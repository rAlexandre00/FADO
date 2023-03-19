from abc import ABC, abstractmethod

from numpy import ndarray


class ServerDefender(ABC):

    def defend_model_parameters(self, clients_model_parameters: list, old_parameters: ndarray) -> list:
        return clients_model_parameters
