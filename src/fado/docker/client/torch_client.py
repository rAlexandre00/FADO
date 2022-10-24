import os
import sys
from importlib import import_module
import torch
import logging
import fedml
from fedml import FedMLRunner
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from fado.data.data_loader import load_partition_data
from fado.logging.prints import HiddenPrints

from fado.security.utils import load_defense_class, load_attack_class

from client_trainer import FadoClientTrainer
from utils import addLoggingLevel, load_yaml_config

_original_stdout = None

def load_data(args):
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
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


def disable_prints():
    global _original_stdout
    _original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')


def enable_prints():
    global _original_stdout
    sys.stdout.close()
    sys.stdout = _original_stdout


if __name__ == "__main__":
    # init FedML framework
    with HiddenPrints():
        args = fedml.init()

    """ 
    If the argument 'attack_spec' is specified, load its contents
    to the main arguments scope
    """
    if hasattr(args, "attack_spec"):

        if '.yaml' in args.attack_spec:
            configuration = load_yaml_config(args.attack_spec)
            for arg_key, arg_val in configuration.items():
                setattr(args, arg_key, arg_val)
        else:
            args.attack_spec = load_attack_class(args)

    """
    Add new logging level to filter out FedML logs
    """
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

    model = fedml.model.create(args, output_dim)

    # Initialize client trainer
    client_trainer = FadoClientTrainer(model, args)

    # start training
    logger.trace("Starting training...")
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer=client_trainer)
    fedml_runner.run()
