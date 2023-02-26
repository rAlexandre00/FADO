import ipaddress
import logging
import os
import pickle
import socket
import struct
import threading
from _thread import start_new_thread
from time import sleep
from typing import List

from fado.constants import SERVER_PORT
from fado.runner.communication.base_com_manager import BaseCommunicationManager
from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.utils import recvall

new_client_lock = threading.Lock()


class ClientSocketCommunicationManager(BaseCommunicationManager):

    def __init__(self, client_id):
        self.client_id = client_id
        self.connections = {}
        self._observers: List[Observer] = []
        self.logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': client_id})

        # This is client -> Connect to server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        base_ip = ipaddress.ip_address('10.128.0.2')
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, str(base_ip + client_id - 1).encode())
        s.connect((os.getenv('SERVER_IP'), SERVER_PORT))
        self.connections[0] = s

        # Store connection
        connect_message = Message(sender_id=client_id, receiver_id=0)
        self.send_message(connect_message)

        self.is_running = True

    def send_message(self, message: Message):
        receiver_id = message.get_receiver_id()
        connection = self.connections[receiver_id]
        message_encoded = pickle.dumps(message)
        connection.sendall(struct.pack('>I', len(message_encoded)))
        connection.sendall(message_encoded)

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        while self.is_running:
            connection = self.connections[0]
            message_size = struct.unpack('>I', recvall(connection, 4))[0]
            message_encoded = recvall(connection, message_size)
            message = pickle.loads(message_encoded)
            for observer in self._observers:
                observer.receive_message(message)

    def stop_receive_message(self):
        self.is_running = False

