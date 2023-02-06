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
from fado.data.data_loader import get_data_loader
from fedml.ml.engine.ml_engine_adapter import get_torch_device

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("fado")
app = Flask(__name__)

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
    data_loader = get_data_loader(args)
    dataset = data_loader.dataset

    logger.info("Data loaded...")

    simulation_datetime = datetime.now()
    board_out = f'runs/{simulation_datetime.strftime("%d.%m.%Y_%H:%M:%S")}'
    os.makedirs(board_out, exist_ok=True)
    writer = SummaryWriter(board_out)

    global server_aggregator
    server_aggregator = FadoServerAggregator(model, writer, args, data_loader.target_test_data)

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    start_server()

    # start training
    try:
        logger.info("Starting training...")
        fedml_runner = FedMLRunner(args, device, dataset, model, server_aggregator=server_aggregator)
        fedml_runner.run()
    finally:
        stop_server()
