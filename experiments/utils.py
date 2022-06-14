import json


def loadHyperParameters():
    config = json.load(open('OptimalHyperParameters.json'))
    print(config)
    return config


def loadFederatedParameters():
    config = json.load(open('FederatedParameters.json'))
    return config
