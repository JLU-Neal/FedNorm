import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from experiments.distributed.moleculenet.main_fedavg import load_data, add_args
from experiments.utils import setOptimalParams
from model.moleculenet.sage_readout import GraphSage, SageMoleculeNet
from FedCovariateShiftEstimating.SetNet_cls import SetNet, get_loss

def main():
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # adopt the optimal settings
    args = setOptimalParams(args)
    # args.data_dir = "./data/moleculenet"
    args.SetNet=False


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
    graph_model.eval()
    set_net.train()
    CSE_optimizer = torch.optim.Adam(list(set_net.parameters()), lr=1e-4)
    CSE_criterion = get_loss().to(device)
    for epoch in range(1000):
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
                with torch.no_grad():
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
        mean_correct.append(correct)
        CSE_optimizer.step()

main()
