import logging
import os.path
import random

import numpy as np

from fado.runner.data.load.server_data_loader import ServerDataLoader
from fado.cli.arguments.arguments import FADOArguments
from fado.runner.fl.fl_server import FLServer
from fado.runner.output.results import Results


logger = logging.getLogger("fado")

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
    if 'logs_file_name' in args:
        fh = logging.FileHandler(os.path.join(os.getenv('LOG_FILE_PATH'), args.logs_file_name.format(**args.__dict__)))
        logger.addHandler(fh)

    # Set the seed for PRNGs to be equal to the trial index
    seed = args.random_seed
    if args.engine == 'pytorch':
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif args.engine == 'pytorch':
        import tensorflow as tf

        tf.random.set_seed(seed)

    np.random.seed(seed)
    random.seed(seed)

    start_server()
