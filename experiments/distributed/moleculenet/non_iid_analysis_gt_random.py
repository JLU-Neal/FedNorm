import argparse
from audioop import avg
from cmath import log
from typing import Tuple
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm, kurtosis, ttest_1samp

import random

from sklearn.exceptions import EfficiencyWarning


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))


from experiments.distributed.moleculenet.non_iid_analysis import from_matrices_to_graphs
from data_preprocessing.molecule.data_loader import *
from experiments.distributed.moleculenet.main_fedavg import add_args



def generate_random_graph(graphs: list)->list:
    generated_graphs = []
    for graph in graphs:
        generated_graph = nx.gnm_random_graph(graph.number_of_nodes(), graph.number_of_edges())
        generated_graphs.append(generated_graph)


    return generated_graphs






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)


    path = args.data_dir + args.dataset
    adj_matrices, feature_matrices, labels = get_data(path)
    graphs = from_matrices_to_graphs(adj_matrices)
    random_graphs = generate_random_graph(graphs)
    pass