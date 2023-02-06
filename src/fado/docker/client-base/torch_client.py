import os
import logging
import fedml
from fedml import FedMLRunner
from fado.data.data_loader import get_data_loader
from fado.logging.prints import HiddenPrints
from fado.models import get_model

from fado.security.utils import load_attack
from fedml.ml.engine.ml_engine_adapter import get_torch_device

from client_trainer import FadoClientTrainer

logger = logging.getLogger("fado")


if __name__ == "__main__":
    # init FedML framework
    with HiddenPrints():
        args = fedml.init()

    load_attack(args, 'client_attack_spec')

    fh = logging.FileHandler(os.path.join(f'logs/client_{args.rank}.log'))
    logger.addHandler(fh)

    device = get_torch_device(args, args.using_gpu, 0, "gpu")

    # Get the model
    model = get_model(args)

    # load data
    data_loader = get_data_loader(args)
    dataset = data_loader.dataset

    logger.info("Data loaded...")
    # Initialize client trainer
    client_trainer = FadoClientTrainer(model, args)

    # start training
    logger.info("Starting training...")
    fedml_runner = FedMLRunner(args, device, dataset, model, client_trainer=client_trainer)
    fedml_runner.run()
