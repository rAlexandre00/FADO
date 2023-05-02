import gzip
import logging
import pickle
import select
import socket
import struct
import sys
import threading
import traceback
from multiprocessing import Process
from time import sleep
from typing import List, Optional, Any

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_IP, SERVER_PORT
from fado.runner.communication.base_com_manager import BaseCommunicationManager
from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.utils import recvall

logger = logging.getLogger("fado")
extra = {'node_id': 'server'}
logger = logging.LoggerAdapter(logger, extra)

TCP_USER_TIMEOUT = 18

fado_args = FADOArguments()


class ServerSocketCommunicationManager(BaseCommunicationManager):

    def __init__(self):
        self.id = 0
        self.connections = {}
        self._observers: List[Observer] = []

        # This is server -> listen for client connections
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', SERVER_PORT))
        self.server_socket.listen()
        self.is_running = True

        # Wait for client connections and store them in connections
        threading.Thread(target=self.accept_clients_loop, args=()).start()

    def accept_clients_loop(self):
        try:
            while self.is_running:
                # establish connection with client
                c, addr = self.server_socket.accept()
                try:
                    threading.Thread(target=self.register_new_client, args=(c,), daemon=True).start()
                except Exception:
                    logger.error(traceback.format_exc())
                    pass
        except Exception as e:
            if self.is_running:
                logger.error(traceback.format_exc())
                raise

    def register_new_client(self, connection):
        try:
            message_size = struct.unpack('>I', recvall(connection, 4))[0]
            message_compressed = recvall(connection, message_size)
            message_encoded = gzip.decompress(message_compressed)
            connect_message = pickle.loads(message_encoded)

            # lock acquired by client
            self.connections[connect_message.sender_id] = connection
            # logger.info(f"Client {connect_message.sender_id} connected")
            for observer in self._observers:
                observer.receive_message(connect_message)
        except TypeError:
            pass  # ignore. Heartbeat message

    def send_message(self, message: Message):
        receiver_id = message.get_receiver_id()
        connection = self.connections[receiver_id]
        message_encoded = pickle.dumps(message)
        message_compressed = gzip.compress(message_encoded)
        try:
            connection.settimeout(fado_args.wait_for_clients_timeout)
            connection.sendall(struct.pack('>I', len(message_compressed)))
            connection.sendall(message_compressed)
            return True
        except Exception as e:
            logger.info(f"Client {receiver_id} did not reply. Skipping contribution")
            return False

    def receive_message(self, sender_id) -> Optional[Message]:
        """ Tries to receive a message for 1 second

        :param sender_id:
        :return:
        """
        connection = self.connections[sender_id]
        try:
            connection.settimeout(fado_args.wait_for_clients_timeout)
            message_size = struct.unpack('>I', recvall(connection, 4))[0]
            message_compressed = recvall(connection, message_size)
            message_encoded = gzip.decompress(message_compressed)
            message = pickle.loads(message_encoded)
            return message
        except (socket.timeout, ConnectionResetError):
            logger.info(f"Client {sender_id} did not reply. Skipping contribution")
            return None

    def get_available_clients(self):
        return len(self.connections)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        raise Exception("Method not implemented")

    def stop_receive_message(self):
        logger.info("Closing server clients socket")
        self.is_running = False
        for c in self.connections.values():
            c.close()
        self.server_socket.shutdown(2)
        self.server_socket.close()
