from fado.runner.communication.base_com_manager import BaseCommunicationManager
from fado.runner.communication.message import Message


class FLCommunicationManager(BaseCommunicationManager):

    def __init__(self, id):
        self.id = id

    def send_message(self, msg: Message):
        pass

    def receive_message(self, msg_type, msg_params) -> None:
        pass

    def handle_receive_message(self):
        pass

    def stop_receive_message(self):
        pass
