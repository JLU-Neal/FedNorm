import argparse
from audioop import avg
from cmath import log
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm, kurtosis, ttest_1samp
import logging
import random

from sklearn.exceptions import EfficiencyWarning

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

def analysis_degree_distribution(graphs: list, popmean: float):
    # kurtosis of degree distribution
    overall_hist, _ = degree_distribution(graphs)
    kur = kurtosis(overall_hist)

    #  t-test for degree distribution
    sample_observation = [[idx] * int(overall_hist[idx]) for idx in range(overall_hist.size)]
    sample_observation = [item for sublist in sample_observation for item in sublist]
    stattistic, p_value = ttest_1samp(sample_observation, popmean)

    return kur, p_value

def avg_shortest_path_length(graphs: list):
    avg_length_each_graph = []
    for graph in graphs:
        sum_avg_path = 0
        sum_weight = 0
        for C in (graph.subgraph(c).copy() for c in nx.connected_components(graph)):
            ele_num = len(C.nodes)
            #  weight is calculated based on the number of paths 
            #  between every pair of two node in a component
            weight = (1+ele_num) * ele_num / 2
            sum_weight += weight
            sum_avg_path += nx.average_shortest_path_length(C) * weight

        avg_shortest_path_length = sum_avg_path / sum_weight
        avg_length_each_graph.append(avg_shortest_path_length)

    avg_len = sum(avg_length_each_graph) / len(avg_length_each_graph)

    return avg_length_each_graph, avg_len
    
        
def analysis_avg_shortest_path_length(graphs: list, popmean: float):
    avg_length_each_graph, avg_len = avg_shortest_path_length(graphs)
    
    # t-test for avg shortest path length
    stattistic, p_value = ttest_1samp(avg_length_each_graph, popmean)

    return avg_len, p_value


def local_effciency(graphs: list):
    effciency_each_graph = []
    for graph in graphs:
        effciency_each_graph.append(nx.local_efficiency(graph))

    avg_effciency = sum(effciency_each_graph) / len(effciency_each_graph)

    return effciency_each_graph, avg_effciency

def analysis_local_efficiency(graphs: list, popmean: float):
    effciency_each_graph, avg_effciency = local_effciency(graphs)
    
    # t-test for local efficiency
    stattistic, p_value = ttest_1samp(effciency_each_graph, popmean)

    return avg_effciency, p_value


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


    # popmean_degree = degree_distribution(graphs)[1]
    # popmean_avg_shortest_path_length = avg_shortest_path_length(graphs)[1]
    popmean_local_efficiency = local_effciency(graphs)[1]
    for idx in range(chunk_num):
        graphs_sublist = graphs_chunked[idx]

        # kur, p_value_deg_dis = analysis_degree_distribution(graphs_sublist, popmean_degree)
        # print("kurtosis: {}, p-value: {}".format(kur, p_value_deg_dis))

        # avg_path_len, p_value_avg_len = analysis_avg_shortest_path_length(graphs_sublist, popmean_avg_shortest_path_length)
        # print("avg shortest path length: {}, p-value: {}".format(avg_path_len, p_value_avg_len))
    
        avg_effciency, p_value_local_eff = analysis_local_efficiency(graphs_sublist, popmean_local_efficiency)
        print("avg local efficiency: {}, p-value: {}".format(avg_effciency, p_value_local_eff))
    pass

