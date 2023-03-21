import logging

import numpy as np
from numpy import ndarray

from fado.cli.arguments.arguments import FADOArguments
from fado.security.defend.server.server_defense_base import ServerDefender

fado_args = FADOArguments()
logger = logging.getLogger("fado")
extra = {'node_id': 'server'}
logger = logging.LoggerAdapter(logger, extra)


class ClippingDefender(ServerDefender):

    def defend_model_parameters(self, clients_model_parameters: list, old_parameters: ndarray) -> list:
        # Compute the 2-norm of the difference between the old and new weights for
        # each client.
        norm_li = [np.linalg.norm([np.linalg.norm(
            old - w) for (old, w) in zip(old_parameters, user)]) for user in clients_model_parameters]

        # Clip the norms
        norm_li = [max(1, norm / fado_args.clip_norm) for norm in norm_li]

        # Divide the updated by the clipped norms
        clipped_client_parameters = [[(w - old) / norm + old for (old, w) in zip(old_parameters, user)]
                                     for (user, norm) in zip(clients_model_parameters, norm_li)]

        clipped_client_parameters = [np.array(parameters, dtype=object) for parameters in clipped_client_parameters]

        return clipped_client_parameters
