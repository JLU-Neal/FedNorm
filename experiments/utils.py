import json


def loadHyperParameters():
    config = json.load(open('../../OptimalHyperParameters.json'))
    return config


def loadFederatedParameters():
    config = json.load(open('../../FederatedParameters.json'))
    return config



def setOptimalParams(args):
    hyperParamsConfig = loadHyperParameters()[args.model + ' + ' + args.fl_algorithm.split("_")[0] + ' on ' + args.dataset]
    fedParamsConfig = loadFederatedParameters()

    # model params
    args.lr = hyperParamsConfig['learning rate']
    args.dropout = hyperParamsConfig['dropout rate']
    args.node_embedding_dim = hyperParamsConfig['node embedding dimension']
    args.hidden_size = hyperParamsConfig['hidden layer dimension']
    args.readout_hidden_dim = hyperParamsConfig['readout embedding dimension']
    args.graph_embedding_dim = hyperParamsConfig['graph embedding dimension']
    args.num_heads = hyperParamsConfig['attention heads']
    args.alpha = hyperParamsConfig['alpha']
    args.partition_alpha = hyperParamsConfig['PARTITION_ALPHA'] * args.p_alpha_factor

    # fed params
    args.client_num_in_total = fedParamsConfig['CLIENT_NUM']
    # args.client_num_per_round = fedParamsConfig['WORKER_NUM']
    args.gpu_server_num = fedParamsConfig['SERVER_NUM']
    args.gpu_num_per_server = fedParamsConfig['GPU_NUM_PER_SERVER']
    args.comm_round = fedParamsConfig['ROUND']
    args.epochs = fedParamsConfig['EPOCH']
    args.batch_size = fedParamsConfig['BATCH_SIZE']
    args.cse_lr = fedParamsConfig['CSE_LEARNING_RATE']

    return args