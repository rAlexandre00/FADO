import logging

from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.client_com_manager import ClientSocketCommunicationManager
from fado.runner.ml.model.module_manager import ModelManager

logger = logging.getLogger("fado")


class FLClient(Observer):
    """ Class representing a server in the federated learning protocol
    """

    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.local_model = ModelManager.get_model()
        self.com_manager = ClientSocketCommunicationManager(client_id=client_id)
        # Add FLServer to observers in order to receive notification of new clients
        self.com_manager.add_observer(self)
        self.dataset = dataset
        self.logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': client_id})

    def start(self):
        self.com_manager.handle_receive_message()

    def stop(self):
        self.com_manager.stop_receive_message()

    def receive_message(self, message) -> None:
        """ Called when receives message

        :param message:
        :return:
        """
        if message.get_type() == Message.MSG_TYPE_END:
            self.stop()
            self.logger.info(f'Received stop message')
        elif message.get_type() == Message.MSG_TYPE_SEND_MODEL:
            self.logger.info(f'Received model and starting training')
            received_parameters = message.get(Message.MSG_ARG_KEY_MODEL_PARAMS)
            self.local_model.set_parameters(received_parameters)
            self.local_model.train(self.dataset.train_data['x'], self.dataset.train_data['y'])

            result_message = Message(type=Message.MSG_TYPE_SEND_MODEL, sender_id=self.client_id, receiver_id=0)
            result_message.add(Message.MSG_ARG_KEY_MODEL_PARAMS, self.local_model.get_parameters())
            self.logger.info(f'Training done and sending resulting model parameters')
            self.com_manager.send_message(result_message)
            self.logger.info(f'Local model parameters sent')
        else:
            self.logger.error(f"Unknown message type received: {message.get_type()}")


