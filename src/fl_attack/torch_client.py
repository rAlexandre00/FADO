import fedml
import logging
from fedml import FedMLRunner
from fedml.core.mlops.mlops_runtime_log import MLOpsRuntimeLog
from client_trainer import ClientTrainerAFAF
from utils import addLoggingLevel, load_yaml_config

if __name__ == "__main__":
    args = fedml.init()

    if hasattr(args, "attack_spec"):
        configuration = load_yaml_config(args.attack_spec)
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
    logger.trace("Loading data...")
    dataset, output_dim = fedml.data.load(args)
    logger.trace("Data loaded...")

    # load model
    model = fedml.model.create(args, output_dim)

    client_trainer = ClientTrainerAFAF(model, args)

    # start training
    logger.trace("Starting training...")
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer=client_trainer)
    fedml_runner.run()
