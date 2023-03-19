from abc import ABC, abstractmethod
from typing import Callable

from numpy import ndarray


class Attack(ABC):
    def attack_model_parameters(self, model_parameters: ndarray, old_parameters: ndarray) -> ndarray:
        return model_parameters
