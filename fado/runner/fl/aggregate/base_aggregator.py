from abc import abstractmethod

from numpy import ndarray

from fado.runner.ml.model.fado_module import FADOModule


class Aggregator:

    def __init__(self, global_model: FADOModule):
        self.model = global_model

    @abstractmethod
    def aggregate(self, parameters) -> ndarray:
        pass
