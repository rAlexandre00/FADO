import gzip
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
connections_list_lock = threading.Lock()


def receive_message(connection) -> Optional[Message]:
    """ Tries to receive a message for 1 second

    :param connection:
    :return:
    """
    ready = select.select([connection], [], [], 1)
    if ready[0]:
        message_size = struct.unpack('>I', recvall(connection, 4))[0]
        message_compressed = recvall(connection, message_size)
        message_encoded = gzip.decompress(message_compressed)
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
        self.reply_connections = []

        threading.Thread(target=self.reply_loop, args=(), daemon=True).start()

    def reply_loop(self):
        while self.is_running:
            # establish connection with client
            try:
                c, addr = self.server_socket.accept()
            except Exception as e:
                if self.is_running:
                    raise e
                else:
                    break
            message = receive_message(c)
            connections_list_lock.acquire()
            self.reply_connections.append(c)
            connections_list_lock.release()
            if message:
                for observer in self._observers:
                    observer.receive_message(message)

    def send_message(self, message: Message):
        connections_list_lock.acquire()
        connection = self.reply_connections.pop()
        connections_list_lock.release()
        message_encoded = pickle.dumps(message)
        message_compressed = gzip.compress(message_encoded)
        connection.sendall(struct.pack('>I', len(message_compressed)))
        connection.sendall(message_compressed)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        raise Exception("Method not implemented")

    def stop_receive_message(self):
        logger.info("Closing server pub socket")
        self.is_running = False
        self.server_socket.shutdown(2)
        self.server_socket.close()
