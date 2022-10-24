import logging
import random
import fedml
import numpy as np
import torch
from fado.security.attack import ModelAttack
from fedml.core.security.common.utils import is_weight_param
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ByzantineAttack(ModelAttack):
    def __init__(self, args):
        super().__init__()
        self.attack_mode = args.attack_mode  # random: randomly generate a weight; zero: set the weight to 0
        self.device = fedml.device.get_device(args)

    def attack_model(self, raw_client_grad: Tuple[float, Dict],
        extra_auxiliary_info: Any = None):
        
        if self.attack_mode == "zero":
            byzantine_local_w = self._attack_zero_mode(raw_client_grad)
        elif self.attack_mode == "random":
            byzantine_local_w = self._attack_random_mode(raw_client_grad)
        else:
            raise NotImplementedError("Method not implemented!")

        return byzantine_local_w

    def _attack_zero_mode(self, model_params):
        
        for k in model_params.keys():
            if is_weight_param(k):
                model_params[k] = torch.from_numpy(np.zeros(model_params[k].size())).float().to(self.device)
        return model_params

    def _attack_random_mode(self, model_params):

        for k in model_params.keys():
            if is_weight_param(k):
                model_params[k] = torch.from_numpy(np.random.random_sample(size=model_params[k].size())).float().to(self.device)
        return model_params