import fedml
import fedml.cross_silo.server.fedml_aggregator

from fado.fedml_diff.core.distributed.fedml_comm_manager import _init_manager
from fado.fedml_diff.cross_silo.server.fedml_aggregator import data_silo_selection

fedml.core.distributed.fedml_comm_manager.FedMLCommManager._init_manager = _init_manager
fedml.cross_silo.server.fedml_aggregator.FedMLAggregator.data_silo_selection = data_silo_selection
