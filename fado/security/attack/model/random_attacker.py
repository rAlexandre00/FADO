
import logging
from numpy import ndarray

from fado.cli.arguments.arguments import FADOArguments
from fado.security.attack.attack_base import Attack

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'router'})
fado_args = FADOArguments("/app/config/fado_config.yaml")


class RandomAttacker(Attack):

    def attack_model_parameters(self, model_parameters: ndarray) -> ndarray:
        logger.info(model_parameters)
        return model_parameters


    