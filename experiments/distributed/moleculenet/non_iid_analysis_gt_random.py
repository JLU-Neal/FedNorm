import argparse
from audioop import avg
from cmath import log
from typing import Tuple
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm, kurtosis, ttest_ind

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


def list_avgDegree(graphs: list)->list:
    avgDegrees = []
    for graph in graphs:
        degree_hist = nx.degree_histogram(graph)
        # calculate the average degree
        sum_degree = 0
        count_degree = 0
        for i in range(len(degree_hist)):
            sum_degree += degree_hist[i] * i
            count_degree += degree_hist[i]
        avgDegree = sum_degree / count_degree
        avgDegrees.append(avgDegree)
    return avgDegrees


def list_shortestPathLength(graphs: list)->list:
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

    return avg_length_each_graph


def list_localEfficiency(graphs: list)->list:
    local_efficiencies = []
    for graph in graphs:
        local_efficiencies.append(nx.local_efficiency(graph))
    return local_efficiencies



def _get_p_val(graphs_gt: list, graphs_random: list, func: callable):
    samples_gt = func(graphs_gt)
    samples_random = func(graphs_random)
    p_val = ttest_ind(samples_gt, samples_random).pvalue

    return p_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)


    path = args.data_dir + args.dataset
    adj_matrices, feature_matrices, labels = get_data(path)
    graphs = from_matrices_to_graphs(adj_matrices)
    random_graphs = generate_random_graph(graphs)

    p_val_avgDegree = _get_p_val(graphs, random_graphs, list_avgDegree)

    p_val_shortestPathLength = _get_p_val(graphs, random_graphs, list_shortestPathLength)
    
    p_val_localEfficiency = _get_p_val(graphs, random_graphs, list_localEfficiency)
    

    print("p_val_avgDegree: ", p_val_avgDegree)
    print("p_val_shortestPathLength: ", p_val_shortestPathLength)
    print("p_val_localEfficiency: ", p_val_localEfficiency)
    pass