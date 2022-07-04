import os
import sys
import pickle
import logging


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
import experiments.experiments_manager as experiments_manager
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./experiments_manager.py")))


if os.path.exists("./experiment_manager.pkl"):
    with open("./experiment_manager.pkl", "rb") as f:
        em = pickle.load(f)
    logging.info("experiment_manager.pkl is loaded")