class Experiment:
    def __init__(self):
        self.Model = None
        self.FedAlgorithm = None
        self.Dataset = None
        self.Time = None
        self.performance_by_iterations = []

    def setAttr(self, Model, FedAlgorithm, Dataset, Time):
        self.Model = Model
        self.FedAlgorithm = FedAlgorithm
        self.Dataset = Dataset
        self.Time = Time

class ExperimentsManager:
    def __init__(self):
        self.experiments = {}

experiment = Experiment()