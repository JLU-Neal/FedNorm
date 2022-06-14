import json


def loadHyperParameters():
    config = json.load(open('../../OptimalHyperParameters.json'))
    print(config)
    return config


def loadFederatedParameters():
    config = json.load(open('../../FederatedParameters.json'))
    return config



def setOptimalParams(args):
    hyperParamsConfig = loadHyperParameters()[args.model + ' + ' + args.fl_algorithm + ' on ' + args.dataset]
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

    # fed params
    args.client_num_in_total = fedParamsConfig['CLIENT_NUM']
    args.client_num_per_round = fedParamsConfig['WORKER_NUM']
    args.gpu_server_num = fedParamsConfig['SERVER_NUM']
    args.gpu_num_per_server = fedParamsConfig['GPU_NUM_PER_SERVER']
    args.partition_alpha = fedParamsConfig['PARTITION_ALPHA']
    args.comm_round = fedParamsConfig['ROUND']
    args.epochs = fedParamsConfig['EPOCH']
    args.batch_size = fedParamsConfig['BATCH_SIZE']

    return args