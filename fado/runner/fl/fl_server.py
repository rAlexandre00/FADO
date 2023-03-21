import gzip
import logging
import sys
import threading
import time
from _thread import start_new_thread

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.server_com_manager import ServerSocketCommunicationManager
from fado.runner.communication.sockets.server_pub_info_manager import ServerSocketPubInfoManager
from fado.runner.fl.aggregate.aggregator_manager import AggregatorManager
from fado.runner.fl.select.participant_selector_manager import ParticipantSelectorManager
from fado.runner.ml.model.module_manager import ModelManager
from fado.runner.output.results import Results
from fado.security.defend.server.server_defense_manager import ServerDefenseManager

logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': 'server'})

fado_args = FADOArguments()
clients_models_dict_lock = threading.Lock()
model_update_lock = threading.Lock()


class FLServer(Observer):
    """ Class representing a server in the federated learning protocol
    """

    def __init__(self, dataset, results):
        self.global_model = ModelManager.get_model()
        self.dataset = dataset
        self.results = results
        self.is_running = False
        self.current_round = 1
        self.com_manager = ServerSocketCommunicationManager()
        self.pub_com_manager = ServerSocketPubInfoManager()
        # Add FLServer to observers in order to receive notification of new clients
        self.com_manager.add_observer(self)
        self.pub_com_manager.add_observer(self)
        self.participant_selector = ParticipantSelectorManager.get_selector()
        self.aggregator = AggregatorManager.get_aggregator(self.global_model)
        self.results = Results()
        self.defender = ServerDefenseManager.get_defender()

    def start(self):
        for self.current_round in range(fado_args.rounds):
            successful = False
            while not successful:
                # Wait for enough clients
                while not self.is_running:
                    time.sleep(0.01)
                successful = self._train_round()

    def stop(self):
        try:
            self.results.write_to_file()
            for client_id in range(1, fado_args.number_clients + 1):
                end_message = Message(type=Message.MSG_TYPE_END, sender_id=0, receiver_id=client_id)
                self.com_manager.send_message(end_message)
        finally:
            self.com_manager.stop_receive_message()
            self.pub_com_manager.stop_receive_message()
            self.is_running = False

    def _train_round(self):
        # 1. Select clients 'num_clients_select' clients (bigger than 'clients_per_round')
        round_clients = self.participant_selector.get_participants(list(range(1, fado_args.number_clients)),
                                                                   fado_args.num_clients_select)
        logger.info(f"Starting round {self.current_round} with clients {round_clients}")

        # 2. Send models and wait for clients_per_round responses
        self.client_models = []
        self.waiting_threads = {}
        for client_id in round_clients:
            self.waiting_threads[client_id] = threading.Thread(target=self._wait_for_client_model,
                                                               args=(client_id, self.current_round), daemon=True)
            self.waiting_threads[client_id].start()
        for client_id in round_clients:
            self.waiting_threads[client_id].join()

        num_models = len(self.client_models)
        if not fado_args.allow_less_clients and num_models < fado_args.clients_per_round:
            return False
        elif num_models == 0:
            return False

        # 3. Aggregate models
        logger.info(f'Aggregating {num_models} models')
        self.client_models = self.defender.defend_model_parameters(self.client_models, self.global_model.get_parameters())
        new_model_parameters = self.aggregator.aggregate(self.client_models)

        # 4. Replace global model
        model_update_lock.acquire()
        self.global_model.set_parameters(new_model_parameters)
        model_update_lock.release()

        # 5. Test new model
        loss, accuracy = self.global_model.evaluate(self.dataset.test_data['x'], self.dataset.test_data['y'])
        logger.info(f'Round loss, accuracy on test data: {loss}, {accuracy}')
        self.results.add_round('per_round_model_accuracy', accuracy)

        loss, accuracy = self.global_model.evaluate(self.dataset.target_test_data['x'],
                                                    self.dataset.target_test_data['y'])
        logger.info(f'Round loss, accuracy on target test data: {loss}, {accuracy}')
        self.results.add_round('per_round_target_accuracy', accuracy)

        return True

    def _send_model(self, client_id):
        send_model_message = Message(type=Message.MSG_TYPE_SEND_MODEL, sender_id=0, receiver_id=client_id)
        send_model_message.add(Message.MSG_ARG_KEY_MODEL_PARAMS, self.global_model.get_parameters())
        return self.com_manager.send_message(send_model_message)

    def _wait_for_client_model(self, client_id, current_round):
        sent = self._send_model(client_id)
        if sent:
            # Try to get response from client
            model_message = self.com_manager.receive_message(client_id)

            # Add the model to the models to be aggregated if the server does not have already enough models
            if model_message is None:
                # Malformed model, ignore
                return
            model = model_message.get(Message.MSG_ARG_KEY_MODEL_PARAMS)
            clients_models_dict_lock.acquire()
            if len(self.client_models) < fado_args.clients_per_round and current_round == self.current_round:
                self.client_models.append(model)
            clients_models_dict_lock.release()
        return

    def receive_message(self, message) -> None:
        """ Called when new clients connect

        :param message: Message
        :return:
        """
        if message.get_type() == message.MSG_TYPE_CONNECT:
            num_clients_available = self.com_manager.get_available_clients()
            logger.info(f"Clients connected {num_clients_available}")
            if not self.is_running and num_clients_available == fado_args.number_clients:
                logger.info("Server has enough clients to start")
                self.is_running = True
        elif message.get_type() == message.MSG_TYPE_GET_MODEL:
            send_model_message = Message(type=Message.MSG_TYPE_SEND_MODEL, sender_id=0, receiver_id=0)
            model_update_lock.acquire()
            send_model_message.add(Message.MSG_ARG_KEY_MODEL_PARAMS, self.global_model.get_parameters())
            model_update_lock.release()
            self.pub_com_manager.send_message(send_model_message)
