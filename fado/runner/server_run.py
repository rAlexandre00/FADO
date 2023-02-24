import os.path
import random

import numpy as np
#import tensorflow as tf
import torch

from fado.runner.data.load.server_data_loader import ServerDataLoader
from fado.cli.arguments.arguments import FADOArguments
from fado.constants import ALL_DATA_FOLDER, FADO_CONFIG_OUT
from fado.runner.fl.fl_server import FLServer
from fado.runner.results.results import Results


def start_server():
    data_path = os.getenv("FADO_DATA_PATH", default='/app/data')
    data_loader = ServerDataLoader(data_path)

    dataset = data_loader.read_data()
    results = Results()
    server = FLServer(dataset=dataset, results=results)
    try:
        server.start()
    finally:
        server.stop()


if __name__ == '__main__':
    # Read arguments to singleton
    args = FADOArguments(os.getenv("FADO_CONFIG_PATH", default="/app/config/fado_config.yaml"))

    # Set the seed for PRNGs to be equal to the trial index
    seed = args.random_seed
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    start_server()
