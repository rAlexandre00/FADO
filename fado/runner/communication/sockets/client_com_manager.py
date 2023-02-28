import ipaddress
import logging
import os
import pickle
import select
import socket
import struct
import threading

from typing import List

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_PORT
from fado.runner.communication.base_com_manager import BaseCommunicationManager
from fado.runner.communication.message import Message
from fado.runner.communication.observer import Observer
from fado.runner.communication.sockets.utils import recvall

fado_args = FADOArguments()
new_client_lock = threading.Lock()


class ClientSocketCommunicationManager(BaseCommunicationManager):

    def __init__(self, client_id):
        self.client_id = client_id
        self.connections = {}
        self._observers: List[Observer] = []
        self.logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': client_id})

        # This is client -> Connect to server
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        base_ip = ipaddress.ip_address('10.128.1.0')
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, str(base_ip + client_id).encode())
        s.connect((os.getenv('SERVER_IP'), SERVER_PORT))
        self.connections[0] = s

        # Store connection
        connect_message = Message(sender_id=client_id, receiver_id=0, type=Message.MSG_TYPE_CONNECT)
        self.send_message(connect_message)

        self.is_running = True

    def send_message(self, message: Message):
        receiver_id = message.get_receiver_id()
        connection = self.connections[receiver_id]
        message_encoded = pickle.dumps(message)
        try:
            connection.settimeout(fado_args.wait_for_clients_timeout)
            connection.sendall(struct.pack('>I', len(message_encoded)))
            connection.sendall(message_encoded)
            connection.settimeout(None)
        except socket.timeout:
            self.logger.info("Could not send model to server")

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        while self.is_running:
            ready = select.select([self.connections[0]], [], [], 1)
            if ready[0]:
                connection = self.connections[0]
                message_size = struct.unpack('>I', recvall(connection, 4))[0]
                message_encoded = recvall(connection, message_size)
                message = pickle.loads(message_encoded)
                for observer in self._observers:
                    observer.receive_message(message)

    def stop_receive_message(self):
        self.is_running = False

