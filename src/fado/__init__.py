import logging

from .orchestrate import prepare_orchestrate

import fedml.cross_silo.server.fedml_aggregator

# Monkey patches
from fado.fedml_diff.core.distributed.fedml_comm_manager import _init_manager
from fado.fedml_diff.core.mlops.mlops_runtime_log import init_logs
from fado.fedml_diff.cross_silo.server.fedml_aggregator import data_silo_selection


fedml.core.distributed.fedml_comm_manager.FedMLCommManager._init_manager = _init_manager
fedml.core.mlops.mlops_runtime_log.MLOpsRuntimeLog.init_logs = init_logs
fedml.core.mlops.MLOpsRuntimeLog.init_logs = init_logs
fedml.cross_silo.server.fedml_aggregator.FedMLAggregator.data_silo_selection = data_silo_selection

# Initialize logger
logger = logging.getLogger("fado")
format_str = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] "
                                   "[%(filename)s:%(lineno)d:%(funcName)s] %("
                                   "message)s",
                               datefmt="%a, %d %b %Y %H:%M:%S")
stdout_handle = logging.StreamHandler()
stdout_handle.setFormatter(format_str)
logger.setLevel(logging.INFO)
logger.addHandler(stdout_handle)
