import os
import sys
import pickle
import logging


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import experiments.experiments_manager as experiments_manager
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./experiments_manager.py")))

date = "26_July.pkl"
if os.path.exists("./experiment_log/"+date):
    with open("./experiment_log/"+date, "rb") as f:
        em = pickle.load(f)
    logging.info("experiment_manager.pkl is loaded")