import logging
import os
import random
import socket
import time
from threading import Thread

import numpy as np
# import tensorflow as tf
import torch

from fado.runner.data.load.client_data_loader import ClientDataLoader
from fado.cli.arguments.arguments import FADOArguments
from fado.constants import ALL_DATA_FOLDER, FADO_CONFIG_OUT, SERVER_PORT, SERVER_IP
from fado.runner.fl.fl_client import FLClient

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'clients'})

def isOpen(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)

    try:
        is_open = s.connect_ex((ip, int(port))) == 0 # True if open, False if not
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
    client = FLClient(client_id=client_id, dataset=dataset)
    client.start()


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

    while not isOpen(SERVER_IP, SERVER_PORT):
        logging.info("Waiting for server to start")
        time.sleep(1)

    for client_id in range(1, args.number_clients+1):
        t = Thread(target=start_client, args=(client_id,), daemon=True)
        t.start()
    t.join()

