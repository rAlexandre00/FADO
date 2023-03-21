import logging
import os
import random
import traceback

import numpy as np

from fado.runner.data.load.server_data_loader import ServerDataLoader
from fado.cli.arguments.arguments import FADOArguments
from fado.runner.fl.fl_server import FLServer
from fado.runner.output.results import Results

logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': 'server'})


def start_server():
    try:
        data_path = os.getenv("FADO_DATA_PATH", default='/app/data')
        data_loader = ServerDataLoader(data_path)

        dataset = data_loader.read_data()
        results = Results()
        server = FLServer(dataset=dataset, results=results)
        server.start()
    except Exception:
        c_logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': 'server'})
        c_logger.error(traceback.format_exc())
    finally:
        logger.info("Stopping server")
        server.stop()


def main():
    # Read arguments to singleton
    args = FADOArguments(os.getenv("FADO_CONFIG_PATH", default="/app/config/fado_config.yaml"))
    if 'logs_file_name' in args:
        log_folder_path = os.getenv("LOG_FILE_PATH", default="/app/logs/")
        log_file_path = os.path.join(log_folder_path, args.logs_file_name.format(**args.__dict__))
        fh = logging.FileHandler(log_file_path)
        logging.getLogger("fado").addHandler(fh)

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
    return


if __name__ == '__main__':
    main()
