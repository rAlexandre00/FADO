import logging
import pickle
import select
import socket
import struct
import sys
import threading
from time import sleep
from typing import List, Optional, Any

from fado.constants import SERVER_IP, SERVER_PORT, SERVER_PUB_PORT
from fado.runner.communication.base_com_manager import BaseCommunicationManager
from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.utils import recvall

logger = logging.getLogger("fado")
extra = {'node_id': 'server'}
logger = logging.LoggerAdapter(logger, extra)

new_client_lock = threading.Lock()


def receive_message(connection) -> Optional[Message]:
    """ Tries to receive a message for 1 second

    :param connection:
    :return:
    """
    ready = select.select([connection], [], [], 1)
    if ready[0]:
        message_size = struct.unpack('>I', recvall(connection, 4))[0]
        message_encoded = recvall(connection, message_size)
        message = pickle.loads(message_encoded)
        return message
    return None


class ServerSocketPubInfoManager(BaseCommunicationManager):

    def __init__(self):
        self.id = 0
        self._observers: List[Observer] = []

        # This is server -> listen for client connections
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        self.server_socket.bind(('0.0.0.0', SERVER_PUB_PORT))
        self.server_socket.listen()
        self.is_running = True
        self.reply_connection = None

        threading.Thread(target=self.reply_loop, args=(), daemon=True).start()

    def reply_loop(self):
        while self.is_running:
            # establish connection with client
            c, addr = self.server_socket.accept()
            message = receive_message(c)
            self.reply_connection = c
            if message:
                for observer in self._observers:
                    observer.receive_message(message)

    def send_message(self, message: Message):
        connection = self.reply_connection
        message_encoded = pickle.dumps(message)
        connection.sendall(struct.pack('>I', len(message_encoded)))
        connection.sendall(message_encoded)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        raise Exception("Method not implemented")

    def stop_receive_message(self):
        self.is_running = False
        self.server_socket.close()
