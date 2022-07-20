import itertools
import subprocess
import logging

# sh run_fedavg_distributed_pytorch.sh graphsage FedAvg sider hetero 3

models = ['graphsage']
fed_algs = ['FedAvg', 'DataSharing', 'FedNorm']
datasets = ['sider']
data_distributions = ['hetero']
client_nums = ['3']

for model, fed_alg, dataset, datadistribution, client_num in list(itertools.product(models, fed_algs, datasets, data_distributions, client_nums)):
    logging.info("===================================================")
    logging.info("Running model: {}".format(model))
    logging.info("Running FedAlg: {}".format(fed_alg))
    logging.info("Running dataset: {}".format(dataset))
    logging.info("Running data distribution: {}".format(datadistribution))
    logging.info("Running client num: {}".format(client_num))
    logging.info("===================================================") 
    if fed_alg == 'FedAvg':   
        subprocess.run("sh run_fedavg_distributed_pytorch.sh".split() + [model, fed_alg, dataset, datadistribution, '0', '0', client_num])
    elif fed_alg == 'DataSharing':
        subprocess.run("sh run_fedavg_distributed_pytorch.sh".split() + [model, fed_alg, dataset, datadistribution, '1', '0', client_num])
    elif fed_alg == 'FedNorm':
        subprocess.run("sh run_fedavg_distributed_pytorch.sh".split() + [model, fed_alg, dataset, datadistribution, '0', '1', client_num])

