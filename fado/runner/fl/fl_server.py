import logging
import sys
import threading
import time
from _thread import start_new_thread

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.server_com_manager import ServerSocketCommunicationManager
from fado.runner.fl.aggregate.aggregator_manager import AggregatorManager
from fado.runner.fl.select.participant_selector_manager import ParticipantSelectorManager
from fado.runner.ml.model.module_manager import ModelManager

logger = logging.getLogger("fado")
extra = {'node_id': 'server'}
logger = logging.LoggerAdapter(logger, extra)

fado_args = FADOArguments()
clients_models_dict_lock = threading.Lock()


class FLServer(Observer):
    """ Class representing a server in the federated learning protocol
    """

    def __init__(self, dataset):
        self.global_model = ModelManager.get_model()
        self.global_model = ModelManager.get_model()
        self.dataset = dataset
        self.is_running = False
        self.current_round = 0
        self.com_manager = ServerSocketCommunicationManager(id=0)
        # Add FLServer to observers in order to receive notification of new clients
        self.com_manager.add_observer(self)
        self.participant_selector = ParticipantSelectorManager.get_selector()
        self.aggregator = AggregatorManager.get_aggregator(self.global_model)

    def start(self):
        for self.current_round in range(fado_args.rounds):
            successful = False
            while not successful:
                # Wait for enough clients
                while not self.is_running:
                    time.sleep(0.01)
                logger.info(f"Starting round {self.current_round}")
                successful = self._train_round()

    def stop(self):
        try:
            for client_id in self.com_manager.get_available_clients():
                end_message = Message(type=Message.MSG_TYPE_END, sender_id=0, receiver_id=client_id)
                self.com_manager.send_message(end_message)
        finally:
            self.com_manager.stop_receive_message()
            self.is_running = False

    def _train_round(self):
        # 1. Select clients 'num_clients_select' clients (bigger than 'clients_per_round')
        clients_available = self.com_manager.get_available_clients()
        round_clients = self.participant_selector.get_participants(clients_available, fado_args.num_clients_select)
        logger.info(f"Starting round {self.current_round} with clients {round_clients}")

        # 2. Send model to clients
        for client_id in round_clients:
            send_model_message = Message(type=Message.MSG_TYPE_SEND_MODEL, sender_id=0, receiver_id=client_id)
            send_model_message.add(Message.MSG_ARG_KEY_MODEL_PARAMS, self.global_model.get_parameters())
            self.com_manager.send_message(send_model_message)

        # 3. Wait for clients_per_round responses (timeout and return False if server has not receive enough models)
        self.client_models = []
        self.waiting_for_models = True
        for client_id in round_clients:
            threading.Thread(target=self._wait_for_client_model, args=(client_id, self.current_round), daemon=True).start()

        initial_time = time.time()
        num_models = 0
        while time.time() - initial_time < fado_args.wait_for_clients_timeout and self.waiting_for_models:
            time.sleep(0.1)
            clients_models_dict_lock.acquire()
            num_models = len(self.client_models)
            if num_models >= fado_args.clients_per_round:
                # Stop waiting for clients
                self.waiting_for_models = False
            clients_models_dict_lock.release()

        if num_models < fado_args.clients_per_round:
            return False
        assert num_models == fado_args.clients_per_round

        # 4. Aggregate models
        new_model_parameters = self.aggregator.aggregate(self.client_models)

        # 5. Replace global model
        self.global_model.set_parameters(new_model_parameters)

        # 6. Test new model
        loss, accuracy = self.global_model.model.evaluate(self.dataset.test_data['x'], self.dataset.test_data['y'])
        logger.info(f'Round loss, accuracy on test data: {loss}, {accuracy}')
        loss, accuracy = self.global_model.model.evaluate(self.dataset.target_test_data['x'], self.dataset.target_test_data['y'])
        logger.info(f'Round loss, accuracy on target test data: {loss}, {accuracy}')

        return True

    def _wait_for_client_model(self, client_id, current_round):
        model_message = None
        # Try to get response from client
        sys.stdout.flush()
        while model_message is None:
            if self.waiting_for_models:
                model_message = self.com_manager.receive_message(client_id)
            else:
                return
        logger.info(f"Received local model from client {client_id}")

        # Add the model to the models to be aggregated if the server does not have already enough models
        model = model_message.get(Message.MSG_ARG_KEY_MODEL_PARAMS)
        clients_models_dict_lock.acquire()
        if len(self.client_models) < fado_args.clients_per_round and current_round == self.current_round:
            self.client_models.append(model)
        clients_models_dict_lock.release()

    def receive_message(self, message) -> None:
        """ Called when new clients connect

        :param message: Message
        :return:
        """
        num_clients_available = len(self.com_manager.get_available_clients())
        logger.info(f"Clients online - {num_clients_available}")
        if not self.is_running and num_clients_available >= fado_args.num_clients_select:
            logger.info("Server has enough clients to start")
            self.is_running = True
        elif self.is_running and num_clients_available < fado_args.num_clients_select:
            logger.info("Server does not have enough clients to start")
            self.is_running = False
