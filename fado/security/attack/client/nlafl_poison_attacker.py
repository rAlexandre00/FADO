
import logging
from numpy import ndarray

from fado.cli.arguments.arguments import FADOArguments
from fado.security.attack.client.attack_base import Attack

logger = logging.getLogger("fado")
fado_args = FADOArguments()


class NLAFLPoisonAttacker(Attack):

    def __init__(self, client_id):
        self.client_id = client_id
        self.poison_count = fado_args.poison_count_multiplier*(fado_args.num_pop_clients//3)
        self.logger = logging.LoggerAdapter(logger, {'node_id': f'{client_id}'})

    def attack_model_parameters(self, model_parameters, old_parameters: ndarray) -> ndarray:
        if (fado_args.num_pop_clients < self.client_id) and \
                (self.client_id <= self.poison_count + fado_args.num_pop_clients):
            self.logger.info("Poisoning model")
            diff = [new - old for (old, new) in zip(old_parameters, model_parameters)]
            new_weights = [d * fado_args.boost_factor +
                           old for (old, d) in zip(old_parameters, diff)]
            return new_weights
        else:
            return model_parameters

