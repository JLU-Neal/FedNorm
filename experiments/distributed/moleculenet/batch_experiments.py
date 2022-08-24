import itertools
import subprocess
import logging

# sh run_fedavg_distributed_pytorch.sh graphsage FedAvg sider hetero 3

models = ['graphsage']

# fed_algs = ['FedAvg', 'FedAvg_DataSharing', 'FedAvg_FedNorm']
fed_algs = ['FedAvg_FedNorm']
# datasets = ['sider', 'bbbp', 'bace', 'Tox21', 'clintox']
datasets = ['Tox21', 'clintox']

data_distributions = ['hetero']
client_nums = ['3'] # should be set to 2 later
p_alpha_factors = ['1']

for model, fed_alg, dataset, datadistribution, client_num, p_alpha_factor in list(itertools.product(models, fed_algs, datasets, data_distributions, client_nums, p_alpha_factors)):
    logging.info("===================================================")
    logging.info("Running model: {}".format(model))
    logging.info("Running FedAlg: {}".format(fed_alg))
    logging.info("Running dataset: {}".format(dataset))
    logging.info("Running data distribution: {}".format(datadistribution))
    logging.info("Running client num: {}".format(client_num))
    logging.info("===================================================") 
    if fed_alg == 'FedAvg':   
        subprocess.run("sh run_fedavg_distributed_pytorch.sh".split() + [model, fed_alg, dataset, datadistribution, 'False', 'False', client_num, p_alpha_factor])
    elif fed_alg == 'FedAvg_DataSharing':
        subprocess.run("sh run_fedavg_distributed_pytorch.sh".split() + [model, fed_alg, dataset, datadistribution, 'True', 'False', client_num, p_alpha_factor])
    elif fed_alg == 'FedAvg_FedNorm':
        subprocess.run("sh run_fedavg_distributed_pytorch.sh".split() + [model, fed_alg, dataset, datadistribution, 'False', 'True', client_num, p_alpha_factor])

