import random

import numpy as np
import torch
import argparse

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_FedAvg_distributed
from FedML.fedml_api.distributed.fedavg_original.FedAvgAPI import FedML_FedAvg_distributed as FedML_FedAvg_distributed_original
from FedML.fedml_api.distributed.fedopt.FedOptAPI import FedML_FedOpt_distributed
from FedML.fedml_api.distributed.fedprox.FedProxAPI import FedML_FedProx_distributed

def get_fl_algorithm_initializer(alg_name, args):
    if alg_name[:6] == "FedAvg":
        if not args.SetNet:
            fl_algorithm = FedML_FedAvg_distributed_original
        else:
            fl_algorithm = FedML_FedAvg_distributed
    elif alg_name == "FedOPT":
        fl_algorithm = FedML_FedOpt_distributed
    elif alg_name == "FedProx":
        fl_algorithm = FedML_FedProx_distributed
    else:
        raise Exception("please do sanity check for this algorithm:"+alg_name)

    return fl_algorithm

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_federated_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # # PipeTransformer related
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # Data related
    parser.add_argument('--partition_method', type=str, default='uniform',
                        help='partition method')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')


    # Learning related
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
                        help='maximum sequence length (default: 128)')

    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
                        help='how many gpus will be used ')

    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')
    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO related
    parser.add_argument('--output_dir', type=str, default="/tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    # Federated Learning related
    parser.add_argument('--fl_algorithm', type=str, default="FedAvg",
                        help='Algorithm list: FedAvg; FedOPT; FedProx ')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--client_num_in_total', type=int, default=6, metavar='NN',
                        help='number of clients in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int,
                        default=4, metavar='NN', help='number of workers')

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1,
                        help='server learning rate (default: 0.001)')

    parser.add_argument('--server_momentum', type=float, default=0,
                        help='server momentum (default: 0)')

    parser.add_argument('--fedprox_mu', type=float, default=1,
                        help='server momentum (default: 1)')
    parser.add_argument(
        '--evaluate_during_training_steps', type=int, default=100, metavar='EP',
        help='the frequency of the evaluation during training')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # GPU device management
    parser.add_argument('--gpu_mapping_file', type=str, default="gpu_mapping.yaml",
                        help='the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str,
                        default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--data_sharing_alpha', type=float, default=0.5)

    parser.add_argument('--data_sharing_beta', type=float, default=0.01)
    
    # parser.add_argument("--SetNet", type=int, default=1, help="Whether to use SetNet")

    parser.add_argument("--epochs_FedCSE", type=int, default=1, help="How many epochs to train FedCSE")

    parser.add_argument("--CSE_aggregate_rounds", type=int, default=2, help="How many rounds to aggregate FedCSE")
    
    parser.add_argument("--CSE_pretrain_rounds", type=int, default=0, help="How many rounds to pretrain FedCSE")

    # parser.add_argument("--is_data_sharing", type=int, default=0, help="Whether to use data sharing")

    parser.add_argument("--SetNet", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument("--is_data_sharing", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")


    parser.add_argument("--cse_lr", type=float, default=0.0015, help="CSE learning rate")

    return parser
