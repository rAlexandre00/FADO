import fedml
from fado.fedml_diff.core.distributed.fedml_comm_manager import _init_manager

fedml.core.distributed.fedml_comm_manager.FedMLCommManager._init_manager = _init_manager