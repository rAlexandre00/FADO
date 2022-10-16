from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from time import sleep

from .constants import (
    ATTACK_TYPE_DATA,
    ATTACK_TYPE_MODEL,
    ATTACK_TYPE_NETWORK
)

class AbstractAttack(ABC):

    def __init__(self, attack_type):
        self.attack_type = attack_type
    def is_model_attack(self):
        return self.attack_type == ATTACK_TYPE_MODEL
    def is_data_attack(self):
        return self.attack_type == ATTACK_TYPE_DATA
    def is_network_attack(self):
        return self.attack_type == ATTACK_TYPE_NETWORK

class DataAttack(AbstractAttack):
    def __init__(self):
        super().__init__(ATTACK_TYPE_DATA)

    @abstractmethod
    def attack_data(self, dataset):
        pass

class ModelAttack(AbstractAttack):
    def __init__(self):
        super().__init__(ATTACK_TYPE_MODEL)

    @abstractmethod
    def attack_model(self, raw_client_grad_list: List[Tuple[float, Dict]],
        extra_auxiliary_info: Any = None):
        pass

class NetworkAttack(AbstractAttack):
    def __init__(self, network_delay):
        super().__init__(ATTACK_TYPE_NETWORK)
        self.network_delay = network_delay

    def attack_network(self):
        sleep(self.network_delay/1000.0)
