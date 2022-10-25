import os
from datetime import datetime

from fado.docker.client.server_aggregator import FadoServerAggregator
from fado.logging.prints import HiddenPrints
from fado.security.utils import load_defense_class
import fedml
import torch
import logging
from fedml import FedMLRunner
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from server_aggregator import FadoServerAggregator
from utils import addLoggingLevel, load_yaml_config
from fado.data.data_loader import load_partition_data

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("fado")


def load_data(args):
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data(
        args,
        args.batch_size,
        train_path=args.data_cache_dir + "/train",
        test_path=args.data_cache_dir + "/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


if __name__ == "__main__":
    # init FedML framework
    with HiddenPrints():
        args = fedml.init()

    """ 
    If the argument 'defense_spec' is specified, load its contents
    to the main arguments scope
    """
    if hasattr(args, "defense_spec"):

        if '.yaml' in args.defense_spec:
            configuration = load_yaml_config(args.defense_spec)
            for arg_key, arg_val in configuration.items():
                setattr(args, arg_key, arg_val)
        else:
            args.defense_spec = load_defense_class(args)


    fh = logging.FileHandler(os.path.join(f'logs/server.log'))
    logger.addHandler(fh)

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    model = fedml.model.create(args, output_dim)

    simulation_datetime = datetime.now()
    board_out = f'runs/{simulation_datetime.strftime("%d.%m.%Y_%H:%M:%S")}'
    os.makedirs(board_out, exist_ok=True)
    writer = SummaryWriter(board_out)

    server_aggregator = FadoServerAggregator(model, writer, args)

    # start training
    logger.info("Starting training...")
    fedml_runner = FedMLRunner(args, device, dataset, model, server_aggregator=server_aggregator)
    fedml_runner.run()
