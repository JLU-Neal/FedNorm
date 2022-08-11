import argparse
from cmath import log
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm, kurtosis
import logging
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data_preprocessing.molecule.data_loader import *
from experiments.distributed.moleculenet.main_fedavg import add_args



def from_matrices_to_graphs(matrices: list) -> list:
    graphs = []
    for matrix in matrices:
        G = nx.from_scipy_sparse_array(matrix)
        graphs.append(G)
    return graphs



def kurtosis_degree_distribution(graphs: list):
    overall_hist = numpy.zeros(10)
    for graph in graphs:
        cur_hist = nx.degree_histogram(graph)
        if overall_hist.size >= len(cur_hist):    
            overall_hist += np.pad(np.asarray(cur_hist), (0, overall_hist.size - len(cur_hist)), 'constant')
        else:
            overall_hist = np.pad(overall_hist, (0, len(cur_hist) - overall_hist.size), 'constant')
            overall_hist += np.asarray(cur_hist)
    kur = kurtosis(overall_hist)
    return kur

def avg_shortest_path_length(graphs: list):
    pass


def visualize_graph(graph: nx.Graph):
    subax1 = plt.subplot(121)
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.savefig("graph.png") 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = add_args(parser)


    path = args.data_dir + args.dataset
    client_number = 4

    adj_matrices, feature_matrices, labels = get_data(path)
    
    graphs = from_matrices_to_graphs(adj_matrices)
    random.seed(666)
    random.shuffle(graphs)
    chunk_num = 10
    chunk_size = len(graphs) // chunk_num
    graphs_chunked = [graphs[i:i+chunk_size] for i in range(0, len(graphs), chunk_size)]

    for idx in range(chunk_num):
        graphs_sublist = graphs_chunked[idx]
        kur = kurtosis_degree_distribution(graphs_sublist)
        print(kur)
    
    pass

