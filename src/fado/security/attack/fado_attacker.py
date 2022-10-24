import logging
from fedml.core.common.ml_engine_backend import MLEngineBackend
from typing import List, Tuple, Dict, Any
from importlib import import_module

logger = logging.getLogger(__name__)

class FadoAttacker:
    _attacker_instance = None

    @staticmethod
    def get_instance():
        if FadoAttacker._attacker_instance is None:
            FadoAttacker._attacker_instance = FadoAttacker()

        return FadoAttacker._attacker_instance

    def __init__(self):
        self.attack_type = None
        self.attacker = None

    def init(self, args):
        if hasattr(args, "attack_spec") and args.attack_spec:
            
            if args.rank == 0: # do not initialize attacker for server
                return

            if isinstance(args.attack_spec , str):
                attack_pkg, attack_module, attack_class = args.attacker_class.split('.')
                self.attacker = getattr(import_module(f'{attack_pkg}.{attack_module}'), f'{attack_class}')(args)
            else:
                self.attacker = args.attack_spec(args)

            logger.trace(f"Initializing attacker! {self.attacker}")

    def is_model_attack(self):
        return self.attacker.is_model_attack() if self.attacker else False

    def is_network_delay_attack(self):
        return self.attacker.is_network_attack() if self.attacker else False

    def is_data_attack(self):
        return self.attacker.is_data_attack() if self.attacker else False

    def attack_model(self, raw_client_grad_list: Tuple[float, Dict], extra_auxiliary_info: Any = None):
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        return self.attacker.attack_model(raw_client_grad_list, extra_auxiliary_info)
    
    def attack_network(self):
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        self.attacker.delay_response()

    def attack_data(self, dataset):
        if self.attacker is None:
            raise Exception("attacker is not initialized!")
        print("attacking data")
        return self.attacker.attack_data(dataset)