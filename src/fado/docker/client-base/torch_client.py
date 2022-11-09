import os
import logging
import fedml
from fedml import FedMLRunner
from fado.data.data_loader import load_partition_data
from fado.logging.prints import HiddenPrints
from fado.models import get_model

from fado.security.utils import load_attack
from fedml.ml.engine.ml_engine_adapter import get_torch_device

from client_trainer import FadoClientTrainer

logger = logging.getLogger("fado")


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
        target_test_data,
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

    load_attack(args, 'client_attack_spec')

    fh = logging.FileHandler(os.path.join(f'logs/client_{args.rank}.log'))
    logger.addHandler(fh)

    device = get_torch_device(args, args.using_gpu, args.rank % 2, "gpu")

    # Get the model
    model = get_model(args)

    # load data
    dataset, output_dim = load_data(args)

    logger.info("Data loaded...")
    # Initialize client trainer
    client_trainer = FadoClientTrainer(model, args)

    # start training
    logger.info("Starting training...")
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer=client_trainer)
    fedml_runner.run()
