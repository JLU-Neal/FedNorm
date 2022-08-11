import argparse
from cmath import log
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm, kurtosis, ttest_1samp
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

def degree_distribution(graphs: list):
    overall_hist = numpy.zeros(10)
    for graph in graphs:
        cur_hist = nx.degree_histogram(graph)
        if overall_hist.size >= len(cur_hist):    
            overall_hist += np.pad(np.asarray(cur_hist), (0, overall_hist.size - len(cur_hist)), 'constant')
        else:
            overall_hist = np.pad(overall_hist, (0, len(cur_hist) - overall_hist.size), 'constant')
            overall_hist += np.asarray(cur_hist)
    
    sum_degree = 0
    count_degree = 0
    for i in range(overall_hist.size):
        sum_degree += overall_hist[i] * i
        count_degree += overall_hist[i]
    avg_degree = sum_degree / count_degree
    return overall_hist, avg_degree

def analysis_degree_distribution(graphs: list, avg_degree: float):
    # kurtosis of degree distribution
    overall_hist, _ = degree_distribution(graphs)
    kur = kurtosis(overall_hist)

    #  t-test for degree distribution
    sample_observation = [[idx] * int(overall_hist[idx]) for idx in range(overall_hist.size)]
    sample_observation = [item for sublist in sample_observation for item in sublist]
    stattistic, p_value = ttest_1samp(sample_observation, avg_degree)

    return kur, p_value

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


    avg_degree = degree_distribution(graphs)[1]
    for idx in range(chunk_num):
        graphs_sublist = graphs_chunked[idx]

        kur, p_value_deg_dis = analysis_degree_distribution(graphs_sublist, avg_degree)
        print("kurtosis: {}, p-value: {}".format(kur, p_value_deg_dis))

        avg_path_len = avg_shortest_path_length(graphs_sublist)
        print(avg_path_len)
    
    pass

