import gzip
import ipaddress
import logging
import os
import pickle
import select
import socket
import struct
import threading
import time
import traceback

from typing import List

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_PORT
from fado.runner.communication.base_com_manager import BaseCommunicationManager
from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.utils import recvall

TCP_USER_TIMEOUT = 18

fado_args = FADOArguments()


class ClientSocketCommunicationManager(BaseCommunicationManager):

    def __init__(self, client_id):
        self.client_id = client_id
        self.connections = {}
        self._observers: List[Observer] = []
        self.logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': client_id})

        # This is client -> Connect to server
        self.create_socket()

        self.is_running = True

    def create_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_ip = os.getenv('SERVER_IP')
        if server_ip != 'localhost':
            base_ip = ipaddress.ip_address('10.128.1.0')
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, str(base_ip + self.client_id).encode())
        try:
            s.connect((server_ip, SERVER_PORT))
            self.connections[0] = s
        except Exception:
            self.logger.error(f'{traceback.format_exc()}')
            self.logger.error("Could not connect to server. Trying again")
            time.sleep(2)
            pass

        # Store connection
        connect_message = Message(sender_id=self.client_id, receiver_id=0, type=Message.MSG_TYPE_CONNECT)
        self.send_message(connect_message)

    def send_message(self, message: Message):
        receiver_id = message.get_receiver_id()
        connection = self.connections[receiver_id]
        connection.setsockopt(socket.IPPROTO_TCP, TCP_USER_TIMEOUT, fado_args.wait_for_clients_timeout * 700)
        message_encoded = pickle.dumps(message)
        message_compressed = gzip.compress(message_encoded)
        try:
            connection.sendall(struct.pack('>I', len(message_compressed)))
            connection.sendall(message_compressed)
        except TimeoutError:
            self.logger.info("Timeout")
            self.create_socket()

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        while self.is_running:
            try:
                connection = self.connections[0]
                message_size = struct.unpack('>I', recvall(connection, 4))[0]
                message_compressed = recvall(connection, message_size)
                message_encoded = gzip.decompress(message_compressed)
                message = pickle.loads(message_encoded)
                for observer in self._observers:
                    observer.receive_message(message)
            except TimeoutError:
                self.create_socket()
            except Exception as e:
                self.create_socket()
                self.logger.error(f'{traceback.format_exc()}')

    def stop_receive_message(self):
        self.connections[0].close()
        self.is_running = False
