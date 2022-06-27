import logging
from re import A
from tkinter import N
from tkinter.messagebox import NO
from turtle import clone

import numpy as np
import torch
# import wandb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm
import copy

from FedML.fedml_core.trainer.model_trainer import ModelTrainer
import experiments.experiments_manager as experiments_manager
from FedCovariateShiftEstimating.SetNet_cls import get_loss, SetNet
from model.moleculenet.sage_readout import GraphSage, SageMoleculeNet

# Trainer for MoleculeNet. The evaluation metric is ROC-AUC


class SageMoleculeNetTrainer(ModelTrainer):
    def __init__(self, model:SageMoleculeNet, args=None):
        super().__init__(model, args)
        
        self.graph_model = copy.deepcopy(model.sage)
        self.setnet = SetNet()
        self.test_data = None


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_cse_params(self):
        return self.graph_model.cpu().state_dict(), self.setnet.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def set_cse_params(self, graphmodel_params, setnet_params):
        logging.info("----------set_cse_params--------")
        self.graph_model.load_state_dict(graphmodel_params)
        self.setnet.load_state_dict(setnet_params)

    def train(self, train_data, device, args, client_index=1):
        if args.SetNet:
            graph_model = self.graph_model
            set_net = self.setnet
            graph_model.to(device)
            set_net.to(device)
            graph_model.train()
            set_net.train()
            CSE_optimizer = torch.optim.Adam(list(set_net.parameters()) + list(graph_model.parameters()), lr=args.lr)
            CSE_criterion = get_loss().to(device)
        model = self.model

        model.to(device)
        model.train()

        test_data = None
        try:
            test_data = self.test_data
        except:
            pass

        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        max_test_score = 0
        best_model_params = {}

        set_feat_to_be_used = None
        latest_graphmodel_params = None
        latest_setnet_params = None

        if args.SetNet:
            for epoch in range(args.epochs_FedCSE):
                graph_feat_list = []
                CSE_optimizer.zero_grad()
                mean_correct = []
                for mol_idxs, (forest, feature_matrix, label, mask) in enumerate(train_data):
                    # Pass on molecules that have no labels
                    if torch.all(mask == 0).item():
                        continue

                    forest = [
                        level.to(device=device, dtype=torch.long, non_blocking=True)
                        for level in forest
                    ]
                    feature_matrix = feature_matrix.to(device=device, dtype=torch.float, non_blocking=True)

                    node_embeddings = graph_model(forest, feature_matrix)
                    # Concat initial node attributed with embeddings from sage
                    graph_feat = torch.cat((feature_matrix, node_embeddings), dim=1) 
                    graph_feat = torch.mean(graph_feat, dim=0).unsqueeze(0)
                    graph_feat_list.append(graph_feat)
                graphs_feat = torch.cat(graph_feat_list, dim=0).unsqueeze(0)
                graphs_feat = graphs_feat.permute(0, 2, 1)
                logits_cse, trans_feat, set_feat = set_net(graphs_feat)
                if set_feat_to_be_used is None:
                    set_feat_to_be_used = set_feat.detach()

                target = torch.tensor([client_index]).to(device)
                loss_cse = CSE_criterion(logits_cse.squeeze(), target.squeeze().long(), trans_feat)
                loss_cse.backward()

                pred_choice = logits_cse.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()    
                logging.info("Epoch: {}, Loss: {}, Correct: {}".format(epoch, loss_cse.item(), correct))
                mean_correct.append(correct)
                CSE_optimizer.step()

            latest_graphmodel_params = {
                k: v.cpu() for k, v in graph_model.state_dict().items()
            }
            latest_setnet_params = {
                k: v.cpu() for k, v in set_net.state_dict().items()
            }
        if args.round_idx < args.CSE_pretrain_rounds:
            pass
        else:
            for epoch in range(args.epochs):
                for mol_idxs, (forest, feature_matrix, label, mask) in enumerate(
                    train_data
                ):
                    # Pass on molecules that have no labels
                    if torch.all(mask == 0).item():
                        continue

                    optimizer.zero_grad()

                    forest = [
                        level.to(device=device, dtype=torch.long, non_blocking=True)
                        for level in forest
                    ]
                    sizes = [level.size() for level in forest]
                    types = [level.dtype for level in forest]

                    feature_matrix = feature_matrix.to(device=device, dtype=torch.float32, non_blocking=True)
                    label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                    mask = mask.to(device=device, dtype=torch.float32, non_blocking=True)
                    set_feat_to_be_used = set_feat_to_be_used.to(device=device, dtype=torch.float32, non_blocking=True)
                    

                    if args.SetNet:
                        logits = model(forest, feature_matrix, set_feat_to_be_used)
                    else:
                        logits = model(forest, feature_matrix)
                    loss = criterion(logits, label) * mask
                    loss = loss.sum() / mask.sum()

                    loss.backward()
                    optimizer.step()

                    if ((mol_idxs + 1) % args.frequency_of_the_test == 0) or (
                        mol_idxs == len(train_data) - 1
                    ):
                        if test_data is not None:
                            test_score, _ = self.test(self.test_data, device, args, set_feat_to_be_used)
                            print(
                                "Epoch = {}, Iter = {}/{}: Test Score = {}".format(
                                    epoch, mol_idxs + 1, len(train_data), test_score
                                )
                            )
                            if test_score > max_test_score:
                                max_test_score = test_score
                                best_model_params = {
                                    k: v.cpu() for k, v in model.state_dict().items()
                                }
                            print("Current best = {}".format(max_test_score))

        return max_test_score, best_model_params, latest_graphmodel_params, latest_setnet_params

    def test(self, test_data, device, args, set_feat_to_be_used=None):
        logging.info("----------test--------")
        model = self.model
        model.eval()
        model.to(device)

        with torch.no_grad():
            y_pred = []
            y_true = []
            masks = []
            for mol_idx, (forest, feature_matrix, label, mask) in enumerate(test_data):
                forest = [
                    level.to(device=device, dtype=torch.long, non_blocking=True)
                    for level in forest
                ]
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )

                if args.SetNet:
                    logits = model(forest, feature_matrix, set_feat_to_be_used)
                else:
                    logits = model(forest, feature_matrix)

                y_pred.append(logits.cpu().numpy())
                y_true.append(label.numpy())
                masks.append(mask.numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        masks = np.array(masks)

        results = []
        for label in range(masks.shape[1]):
            valid_idxs = np.nonzero(masks[:, label])
            truth = y_true[valid_idxs, label].flatten()
            pred = y_pred[valid_idxs, label].flatten()

            if np.all(truth == 0.0) or np.all(truth == 1.0):
                results.append(float("nan"))
            else:
                if args.metric == "prc-auc":
                    precision, recall, _ = precision_recall_curve(truth, pred)
                    score = auc(recall, precision)
                else:
                    score = roc_auc_score(truth, pred)

                results.append(score)

        score = np.nanmean(results)
        return score, model

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        if args.round_idx < args.CSE_pretrain_rounds:
            return True

        logging.info("----------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            set_feat_to_be_used = None
            with torch.no_grad():
                graph_model = self.graph_model
                set_net = self.setnet
                graph_model.eval()
                set_net.eval()
                graph_model.to(device)
                set_net.to(device)
                graph_feat_list = []
                for mol_idxs, (forest, feature_matrix, label, mask) in enumerate(test_data):
                    forest = [
                        level.to(device=device, dtype=torch.long, non_blocking=True)
                        for level in forest
                    ]
                    feature_matrix = feature_matrix.to(device=device, dtype=torch.float, non_blocking=True)
                    node_embeddings = graph_model(forest, feature_matrix)
                    # Concat initial node attributed with embeddings from sage
                    graph_feat = torch.cat((feature_matrix, node_embeddings), dim=1) 
                    graph_feat = torch.mean(graph_feat, dim=0).unsqueeze(0)
                    graph_feat_list.append(graph_feat)
                graphs_feat = torch.cat(graph_feat_list, dim=0).unsqueeze(0)
                logging.info("graphs_feat.shape: {}".format(graphs_feat.shape))
                graphs_feat = graphs_feat.permute(0, 2, 1)
                logits_cse, trans_feat, set_feat = set_net(graphs_feat)
                if set_feat_to_be_used is None:
                    set_feat_to_be_used = set_feat.detach()
               

            
            score, model = self.test(test_data, device, args, set_feat_to_be_used)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            logging.info("Client {}, Test ROC-AUC score = {}".format(client_idx, score))
            # wandb.log({"Client {} Test/ROC-AUC".format(client_idx): score})
        avg_score = np.mean(np.array(score_list))
        logging.info("Test ROC-AUC Score = {}".format(avg_score))
        # wandb.log({"Test/ROC-AUC": avg_score})
        try:
            experiments_manager.experiment.performance_by_iterations.append(avg_score)
            logging.info("Result of current iteration saved")
        except NameError:
            logging.info("NameError found!!!!!")

        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(
            model_1.state_dict().items(), model_2.state_dict().items()
        ):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info("Mismtach found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info("Models match perfectly! :)")
