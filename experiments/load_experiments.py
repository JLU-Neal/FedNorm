import os
import sys
import pickle
import logging
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import experiments.experiments_manager as experiments_manager
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./experiments_manager.py")))

date = "experiment_manager.pkl"
if os.path.exists("./experiment_log/"+date):
    with open("./experiment_log/"+date, "rb") as f:
        em = pickle.load(f)
    logging.info("experiment_manager.pkl is loaded")

    visualize_experiments = True
    if visualize_experiments:
        fedavg_itr = em.experiments['graphsage FedAvg sider'].performance_by_iterations
        datashare_itr = em.experiments['graphsage FedAvg_DataSharing sider'].performance_by_iterations
        fednorm_itr = em.experiments['graphsage FedAvg_FedNorm sider'].performance_by_iterations
        # draw curve based on iterations
        plt.figure()
        plt.plot(fedavg_itr, label='FedAvg')
        plt.plot(datashare_itr, label='FedAvg_DataSharing')
        plt.plot(fednorm_itr, label='FedAvg_FedNorm')
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title('The comparision of the three Federated algorithms in convergence')
        plt.savefig('./experiment_log/FedAvg.png')

