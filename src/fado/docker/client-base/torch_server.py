import os
import threading
from datetime import datetime
from threading import Thread

import flask
from flask import Flask, request
from werkzeug.serving import make_server

from fado.logging.prints import HiddenPrints
from fado.security.utils import load_defense
from fado.models import get_model
import fedml
import logging
from fedml import FedMLRunner
from server_aggregator import FadoServerAggregator
from fado.data.data_loader import load_partition_data
from fedml.ml.engine.ml_engine_adapter import get_torch_device

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("fado")
app = Flask(__name__)


def load_data(args):
    target_test_path = None
    if hasattr(args, 'target_class'):
        target_test_path = args.data_cache_dir + "/target_test"
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
        target_test_path=target_test_path
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
    return dataset, class_num, target_test_data


class ServerThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.server = make_server('0.0.0.0', 8889, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


def start_server():
    global server
    # App routes defined here
    server = ServerThread()
    server.start()


def stop_server():
    global server
    server.shutdown()


@app.route('/global_model', methods=['GET'])
def get_global_model():
    global server_aggregator
    return str(server_aggregator.get_model_params())


if __name__ == "__main__":
    # init FedML framework
    with HiddenPrints():
        args = fedml.init()

    load_defense(args)

    fh = logging.FileHandler(os.path.join(f'logs/server.log'))
    logger.addHandler(fh)

    device = get_torch_device(args, args.using_gpu, 0, "gpu")

    # Get the model
    model = get_model(args)

    # load data
    logger.info("Loading data...")
    dataset, output_dim, target_test_data = load_data(args)

    simulation_datetime = datetime.now()
    board_out = f'runs/{simulation_datetime.strftime("%d.%m.%Y_%H:%M:%S")}'
    os.makedirs(board_out, exist_ok=True)
    writer = SummaryWriter(board_out)

    global server_aggregator
    server_aggregator = FadoServerAggregator(model, writer, args, target_test_data)

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    start_server()

    # start training
    try:
        logger.info("Starting training...")
        fedml_runner = FedMLRunner(args, device, dataset, model, server_aggregator=server_aggregator)
        fedml_runner.run()
    finally:
        stop_server()
