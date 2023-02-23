import logging

import numpy as np
from numpy.core._multiarray_umath import ndarray

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.fl.aggregate.base_aggregator import Aggregator
from fado.runner.ml.model.fado_module import FADOModule

logger = logging.getLogger("fado")
extra = {'node_id': 'server'}
logger = logging.LoggerAdapter(logger, extra)


class MeanAggregator(Aggregator):

    def __init__(self, global_model: FADOModule):
        super().__init__(global_model)
        fado_args = FADOArguments()
        self.lr = fado_args.agg_learning_rate

    def aggregate(self, received_parameters):
        old_weights = self.model.get_parameters()

        ave_weights = np.mean(received_parameters, axis=0)

        # Here new_weights gets automatically converted to a numpy array
        new_weights = old_weights + self.lr * (ave_weights - old_weights)
        return new_weights