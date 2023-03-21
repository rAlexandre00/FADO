import logging

import numpy
import numpy as np

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
        old_weights = numpy.asarray(self.model.get_parameters(), dtype=object)

        ave_weights = np.mean(numpy.asarray(received_parameters, dtype=object), axis=0)

        # Here new_weights gets automatically converted to a numpy array
        new_weights = old_weights + self.lr * (ave_weights - old_weights)
        return new_weights
