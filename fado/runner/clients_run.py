import logging
import os
import random
import socket
import sys
import threading
import time
import traceback
from multiprocessing import Process
from threading import Thread

import numpy as np

from fado.runner.data.load.client_data_loader import ClientDataLoader
from fado.cli.arguments.arguments import FADOArguments
from fado.constants import ALL_DATA_FOLDER, FADO_CONFIG_OUT, SERVER_PORT, SERVER_IP
from fado.runner.fl.fl_client import FLClient

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': f'clients'})


def isOpen(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if ip != 'localhost':
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, '10.128.1.0'.encode())
    s.settimeout(5)

    try:
        is_open = s.connect_ex((ip, int(port))) == 0  # True if open, False if not
        if is_open:
            s.shutdown(socket.SHUT_RDWR)
    except Exception:
        is_open = False

    s.close()
    return is_open


def start_client(client_id):
    data_path = os.getenv("FADO_DATA_PATH", default='/app/data')
    data_loader = ClientDataLoader(data_path, client_id)

    dataset = data_loader.read_data()
    try:
        client = FLClient(client_id=client_id, dataset=dataset)
        client.start()
    except Exception:
        c_logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': client_id})
        c_logger.error(traceback.format_exc())


def main():
    # Read arguments to singleton
    args = FADOArguments(os.getenv("FADO_CONFIG_PATH", default="/app/config/fado_config.yaml"))

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

    while not isOpen(os.getenv('SERVER_IP'), SERVER_PORT):
        logger.info("Waiting for server to start")
        time.sleep(1)

    for client_id in range(1, args.number_clients + 1):
        t = threading.Thread(target=start_client, args=(client_id,), daemon=True)
        t.start()
    t.join()
    return


if __name__ == '__main__':
    main()
