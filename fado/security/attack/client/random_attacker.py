
import logging
import numpy as np
from numpy import ndarray

from fado.cli.arguments.arguments import FADOArguments
from fado.security.attack.client.attack_base import Attack

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'clients'})
fado_args = FADOArguments()


class RandomAttacker(Attack):

    def attack_model_parameters(self, model_parameters, old_parameters: ndarray) -> ndarray:

        #logger.info(model_parameters)
        return model_parameters

