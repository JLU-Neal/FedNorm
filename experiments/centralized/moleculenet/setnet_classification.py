from ensurepip import bootstrap
import sys
import os
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from experiments.distributed.moleculenet.main_fedavg import load_data, add_args
from experiments.utils import setOptimalParams
from model.moleculenet.sage_readout import GraphSage, SageMoleculeNet
from FedCovariateShiftEstimating.SetNet_cls import SetNet, get_loss


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test(use_best_model : bool, seed : int):
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # adopt the optimal settings
    args = setOptimalParams(args)
    # args.data_dir = "./data/moleculenet"
    args.SetNet=True
    
    args.use_best_model = use_best_model

    set_seed(seed)

    dataset, feat_dim, num_cats = load_data(args, args.dataset)
    [
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    ] = dataset

    # load the best model
    try:
        model = SageMoleculeNet(
            feat_dim,
            args.hidden_size,
            args.node_embedding_dim,
            args.dropout,
            args.readout_hidden_dim,
            args.graph_embedding_dim,
            num_cats,
            args
        )
        if args.use_best_model:
            model.load_state_dict(torch.load("./best_graph_sage_model.pt"))
        graph_model = model.sage
    except:
        graph_model = GraphSage(
                    feat_dim,
                    args.hidden_size,
                    args.node_embedding_dim,
                    args.dropout,
                    )


    # for param in graph_model.parameters():
    #     param.requires_grad = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_net = SetNet()
    graph_model.to(device)
    set_net.to(device)
    if args.use_best_model:
        graph_model.eval()
    else:
        graph_model.train()
    set_net.train()
    if args.use_best_model:
        CSE_optimizer = torch.optim.Adam(list(set_net.parameters()), lr=1e-4)
    else:
        CSE_optimizer = torch.optim.Adam(list(set_net.parameters()) + list(graph_model.parameters()), lr=1e-4)
        # CSE_optimizer = torch.optim.Adam(list(set_net.parameters()), lr=1e-4)
    CSE_criterion = get_loss().to(device)
    loss_by_epoch = []
    for epoch in range(100):
        graph_feat_list = []
        CSE_optimizer.zero_grad()
        mean_correct = []
        logits_cse_list = []
        for client_index in range (args.client_num_in_total):
            # print("client_index: ", client_index)
            # graphs_feat_list = []
            
            train_data = train_data_local_dict[client_index]
            for mol_idxs, (forest, feature_matrix, label, mask) in enumerate(train_data):
                # print("mol_idxs: ", mol_idxs)
                # Pass on molecules that have no labels
                if torch.all(mask == 0).item():
                    continue

                forest = [
                    level.to(device=device, dtype=torch.long, non_blocking=True)
                    for level in forest
                ]
                feature_matrix = feature_matrix.to(device=device, dtype=torch.float, non_blocking=True)
                if args.use_best_model:
                    with torch.no_grad():
                        node_embeddings = graph_model(forest, feature_matrix)
                else:
                    node_embeddings = graph_model(forest, feature_matrix)
                # Concat initial node attributed with embeddings from sage
                graph_feat = torch.cat((feature_matrix, node_embeddings), dim=1) 
                graph_feat = torch.mean(graph_feat, dim=0).unsqueeze(0)
                graph_feat_list.append(graph_feat)
            graphs_feat = torch.cat(graph_feat_list, dim=0).unsqueeze(0)
            graphs_feat = graphs_feat.permute(0, 2, 1)
            # print(graphs_feat.shape)
            logits_cse, trans_feat, set_feat = set_net(graphs_feat)
            logits_cse_list.append(logits_cse)

        logits_cse_batch = torch.cat(logits_cse_list, dim=0)
        target = torch.tensor(range (args.client_num_in_total)).to(device)
        loss_cse = CSE_criterion(logits_cse_batch.squeeze(), target.squeeze().long(), trans_feat)
        loss_cse.backward()

        pred_choice = logits_cse_batch.data.max(1)[1]
        print(pred_choice)
        print(target)
        correct = pred_choice.eq(target.long().data).cpu().sum() / len(target)  
        print("Epoch: {}, Loss: {}, Correct: {}".format(epoch, loss_cse.item(), correct))
        loss_by_epoch.append(loss_cse.item())
        mean_correct.append(correct)
        CSE_optimizer.step()
    return loss_by_epoch

if __name__ == "__main__":
    if os.path.exists("./bootstrap_experiments.pkl") and os.path.exists("./non_bootstrap_experiments.pkl"):
        with open("./bootstrap_experiments.pkl", "rb") as f:
            bootstrap_experiments = pickle.load(f)
        logging.info("Loaded bootstrap experiments")
        with open("./non_bootstrap_experiments.pkl", "rb") as f:
            non_bootstrap_experiments = pickle.load(f)
        logging.info("Loaded non_bootstrap experiments")
    else:
        bootstrap_experiments = []
        non_bootstrap_experiments = []
        exp_time= 10
        for i in range(exp_time):
            loss_by_epoch_bootstrap = test(use_best_model=True, seed=i)
            loss_by_epoch_no_bootstrap = test(use_best_model=False, seed=i)
            bootstrap_experiments.append(loss_by_epoch_bootstrap)
            non_bootstrap_experiments.append(loss_by_epoch_no_bootstrap)
            with open("./bootstrap_experiments.pkl", "wb") as f:
                pickle.dump(bootstrap_experiments, f)
            with open("./non_bootstrap_experiments.pkl", "wb") as f:
                pickle.dump(non_bootstrap_experiments, f)
            logging.info("Saved experiments")
    
    bootstrap_experiments = np.array(bootstrap_experiments)
    avg_bootstrap_experiments = np.mean(bootstrap_experiments, axis=0)
    std_bootstrap_experiments = np.std(bootstrap_experiments, axis=0)

    non_bootstrap_experiments = np.array(non_bootstrap_experiments)
    avg_non_bootstrap_experiments = np.mean(non_bootstrap_experiments, axis=0)
    std_non_bootstrap_experiments = np.std(non_bootstrap_experiments, axis=0)

    
    # visualize the loss curve
    plt.clf()
    plt.plot(avg_bootstrap_experiments, label="bootstrap")
    plt.fill_between(range(len(avg_bootstrap_experiments)), avg_bootstrap_experiments - std_bootstrap_experiments, avg_bootstrap_experiments + std_bootstrap_experiments, alpha=0.2)

    plt.plot(avg_non_bootstrap_experiments, label="non-bootstrap")
    plt.fill_between(range(len(avg_non_bootstrap_experiments)), avg_non_bootstrap_experiments - std_non_bootstrap_experiments, avg_non_bootstrap_experiments + std_non_bootstrap_experiments, alpha=0.2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./loss_curve.png")
