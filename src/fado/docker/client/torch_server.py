from fado.docker.client.server_aggregator import FadoServerAggregator
import fedml
import torch
import logging
from fedml import FedMLRunner
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from server_aggregator import FadoServerAggregator
from utils import addLoggingLevel, load_yaml_config
from fado.data.data_loader import load_partition_data


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
    args = fedml.init()

    if hasattr(args, "defense_spec"):
        configuration = load_yaml_config(args.defense_spec)
        for arg_key, arg_val in configuration.items():
            setattr(args, arg_key, arg_val)

    log_file_path, program_prefix = MLOpsRuntimeLog.build_log_file_path(args)

    addLoggingLevel('TRACE', logging.CRITICAL + 5)
    logger = logging.getLogger(log_file_path)
    logger.setLevel("TRACE")
    for handler in logger.handlers:
        handler.setLevel("TRACE") 

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = fedml.model.create(args, output_dim)

    server_aggregator = FadoServerAggregator(model, args)

    # start training
    logger.trace("Starting training...")
    fedml_runner = FedMLRunner(args, device, dataset, model, server_aggregator=server_aggregator)
    fedml_runner.run()
