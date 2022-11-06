import logging

from .orchestrate import prepare_orchestrate

import fedml.cross_silo.server.fedml_aggregator
import fedml.cross_silo.server.fedml_server_manager

# Monkey patches
from fado.fedml_diff.init import update_client_id_list
from fado.fedml_diff.cli.env.collect_env import collect_env
from fado.fedml_diff.core.distributed.fedml_comm_manager import _init_manager
from fado.fedml_diff.core.mlops.mlops_runtime_log import init_logs
from fado.fedml_diff.cross_silo.server.fedml_aggregator import FedMLAggregator
from fado.fedml_diff.cross_silo.server.fedml_server_manager import FedMLServerManager

fedml.update_client_id_list = update_client_id_list
fedml.cli.env.collect_env.collect_env = collect_env
fedml.collect_env = collect_env
fedml.core.distributed.fedml_comm_manager.FedMLCommManager._init_manager = _init_manager
fedml.core.mlops.mlops_runtime_log.MLOpsRuntimeLog.init_logs = init_logs
fedml.core.mlops.MLOpsRuntimeLog.init_logs = init_logs
fedml.cross_silo.server.fedml_aggregator.FedMLAggregator = FedMLAggregator
fedml.cross_silo.server.fedml_server_manager.FedMLServerManager = FedMLServerManager

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
