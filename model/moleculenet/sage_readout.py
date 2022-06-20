import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../")))
from FedCovariateShiftEstimating.SetNet_cls import SetNet

class GraphSage(nn.Module):
    """
    GraphSAGE model (https://arxiv.org/abs/1706.02216) to learn the role of atoms in the molecules inductively.
    Transforms input features into a fixed length embedding in a vector space. The embedding captures the role.
    """

    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GraphSage, self).__init__()

        self.feat_dim = feat_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.layer1 = nn.Linear(2 * feat_dim, hidden_dim1, bias=False)
        self.layer2 = nn.Linear(2 * hidden_dim1, hidden_dim2, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, forest, feature_matrix):
        feat_0 = feature_matrix[forest[0]]  # Of shape torch.Size([|B|, feat_dim])
        feat_1 = feature_matrix[
            forest[1]
        ]  # Of shape torch.size(|B|, fanouts[0], feat_dim)

        # Depth 1
        x = feature_matrix[forest[1]].mean(dim=1)  # Of shape torch.size(|B|, feat_dim)
        feat_0 = torch.cat((feat_0, x), dim=1)  # Of shape torch.size(|B|, 2 * feat_dim)
        feat_0 = self.relu(self.layer1(feat_0))  # Of shape torch.size(|B|, hidden_dim1)
        feat_0 = self.dropout(feat_0)

        # Depth 2
        x = feature_matrix[forest[2]].mean(
            dim=1
        )  # Of shape torch.size(|B|*fanouts[0], feat_dim)
        feat_1 = torch.cat(
            (feat_1.reshape(-1, self.feat_dim), x), dim=1
        )  # Of shape torch.size(|B|*fanouts[0], 2 * feat_dim)
        feat_1 = self.relu(
            self.layer1(feat_1)
        )  # Of shape torch.size(|B|*fanouts[0], hidden_dim1)
        feat_1 = self.dropout(feat_1)

        # Combine
        feat_1 = feat_1.reshape(forest[0].shape[0], -1, self.hidden_dim1).mean(
            dim=1
        )  # Of shape torch.size([|B|, hidden_dim_1])
        combined = torch.cat(
            (feat_0, feat_1), dim=1
        )  # Of shape torch.Size(|B|, 2 * hidden_dim1)
        embeddings = self.relu(
            self.layer2(combined)
        )  # Of shape torch.Size(|B|, hidden_dim2)

        return embeddings


class Readout(nn.Module):
    """
    This module learns a single graph level representation for a molecule given GraphSAGE generated embeddings
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim, num_cats):
        super(Readout, self).__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cats = num_cats

        self.layer1 = nn.Linear(attr_dim + embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Linear(output_dim, num_cats)
        self.act = nn.ReLU()

    def forward(self, node_features, node_embeddings):
        combined_rep = torch.cat(
            (node_features, node_embeddings), dim=1
        )  # Concat initial node attributed with embeddings from sage
        hidden_rep = self.act(self.layer1(combined_rep))
        graph_rep = self.act(
            self.layer2(hidden_rep)
        )  # Generate final graph level embedding

        logits = torch.mean(
            self.output(graph_rep), dim=0
        )  # Generated logits for multilabel classification

        return logits


class Readout_with_SetNet(Readout):
    """
    This module inherited from Readout, with an extra SetNet to learn the covariate shift of current client
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim, num_cats, args=None):
        super(Readout_with_SetNet, self).__init__(attr_dim, embedding_dim, hidden_dim, output_dim, num_cats)
        self.setnet = SetNet()
        x_dim = 1024
        h_dim1 = 128
        h_dim2 = 128
        z_dim = 24
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var

    def transform(self, combined_rep, mu, log_var):
        std = torch.exp(0.5*log_var)
        # eps = torch.randn_like(std)
        return combined_rep.mul(std).add_(mu) # return z sample

    def forward(self, node_features, node_embeddings, set_feat):
        # Concat initial node attributed with embeddings from sage
        combined_rep = torch.cat(
            (node_features, node_embeddings), dim=1
        )  

        # Use set_feat to generate mu and log_var 
        mu, log_var = self.encoder(set_feat)

        # Affine Transformation
        combined_rep = self.transform(combined_rep, mu, log_var)

        # Generate final graph level embedding
        hidden_rep = self.act(self.layer1(combined_rep))
        graph_rep = self.act(
            self.layer2(hidden_rep)
        )   

        # Generated logits for multilabel classification
        logits = torch.mean(
            self.output(graph_rep), dim=0
        )  

        return logits


class SageMoleculeNet(nn.Module):
    """
    Network that consolidates Sage + Readout into a single nn.Module
    """

    def __init__(
        self,
        feat_dim,
        sage_hidden_dim1,
        node_embedding_dim,
        sage_dropout,
        readout_hidden_dim,
        graph_embedding_dim,
        num_categories, args=None
    ):
        super(SageMoleculeNet, self).__init__()
        self.sage = GraphSage(
            feat_dim, sage_hidden_dim1, node_embedding_dim, sage_dropout
        )
        self.readout = Readout(
            feat_dim,
            node_embedding_dim,
            readout_hidden_dim,
            graph_embedding_dim,
            num_categories,
        )

        if args is None or not args.SetNet: 
            self.readout = Readout(
                feat_dim,
                node_embedding_dim,
                readout_hidden_dim,
                graph_embedding_dim,
                num_categories,
            )
        else: 
            self.embedding_for_CSE = GraphSage(
                feat_dim, sage_hidden_dim1, node_embedding_dim, sage_dropout
            )

            self.readout = Readout_with_SetNet(
                feat_dim,
                node_embedding_dim,
                readout_hidden_dim,
                graph_embedding_dim,
                num_categories,
                args
            )

    def forward(self, forest, feature_matrix, set_feat=None):
        node_embeddings = self.sage(forest, feature_matrix)

        if set_feat is None:
            logits = self.readout(feature_matrix, node_embeddings)
        else:
            logits = self.readout(feature_matrix, node_embeddings, set_feat)
        return logits
        # if not isinstance(self.readout, Readout_with_SetNet):
        #     logits = self.readout(feature_matrix, node_embeddings)
        #     return logits
        # else:
        #     embedding_for_CSE = self.embedding_for_CSE(forest, feature_matrix)
        #     logits, logits_cse = self.readout(feature_matrix, node_embeddings, embedding_for_CSE)
        #     return logits, logits_cse
