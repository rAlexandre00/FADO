import gzip
import heapq
import ipaddress
import logging
import os
import pickle
import socket
import struct
import threading
import time
from threading import Lock

import numpy as np

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_PUB_PORT
from fado.runner.communication.message import Message
from fado.runner.communication.sockets.utils import recvall

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'router'})
fado_args = FADOArguments("/app/config/fado_config.yaml")

BASE_IP = ipaddress.ip_address('10.128.1.0')

clients_training_lock = Lock()


def get_model_parameters():
    # Connect

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect((os.getenv('SERVER_IP'), SERVER_PUB_PORT))

    # Send model request
    message = Message(sender_id=-1, receiver_id=-1, type=Message.MSG_TYPE_GET_MODEL)
    message_encoded = pickle.dumps(message)
    s.sendall(struct.pack('>I', len(message_encoded)))
    s.sendall(message_encoded)

    # Receive model parameters
    message_size = struct.unpack('>I', recvall(s, 4))[0]
    message_compressed = recvall(s, message_size)
    message_encoded = gzip.decompress(message_compressed)
    message = pickle.loads(message_encoded)
    return message.get(Message.MSG_ARG_KEY_MODEL_PARAMS)


def check_param_equality(current_model_parameters, old_model_parameters):
    for current_layer, old_layer in zip(current_model_parameters, old_model_parameters):
        if not np.array_equal(current_layer, old_layer, equal_nan=True):
            return False
    return True


class NLAFLAttacker:

    def __init__(self, model, x_target_test, y_target_test):
        """ Called when receives message

        :param model (FADOModule):
        :return:
        """
        logger.info('Starting network attack')
        self.drop_count = fado_args.drop_count_multiplier * (fado_args.num_pop_clients // 3)
        self.current_round = 0
        self.clients_training = []
        # Dict with IPs as key and tuples (sum_perf, count_perf) as values to allow mean calculations
        self.clients_improv_history = {}
        self.clients_prev_round = []
        self.ips_lowest_losses = []
        self.server_is_aggregating = True
        self.local_model = model
        self.x_target_test = x_target_test
        self.y_target_test = y_target_test
        self.last_loss = None
        start_ip = ipaddress.IPv4Address('10.128.1.1')
        target_clients = []
        for ip_int in range(int(start_ip), int(start_ip) + fado_args.num_pop_clients):
            target_clients.append(str(ipaddress.IPv4Address(ip_int)))
        self.target_clients = set(target_clients)
        threading.Thread(target=self.contribution_estimation, args=(), daemon=True).start()

    def contribution_estimation(self):
        while True:
            try:
                old_model_parameters = get_model_parameters()
                logger.info("Got old_model_parameters")
            except ConnectionRefusedError:
                time.sleep(0.5)
                continue
            break

        while True:
            try:
                current_model_parameters = get_model_parameters()
                logger.info("Got current_model_parameters")
            except socket.timeout:
                time.sleep(1)

            # Check new round
            if not check_param_equality(current_model_parameters, old_model_parameters):
                clients_training_lock.acquire()
                logger.info(f"Round {self.current_round} end detected. Estimating clients contribution")
                self.current_round += 1
                self.clients_prev_round = self.clients_training
                self.clients_training = []
                clients_training_lock.release()
                old_model_parameters = current_model_parameters
                self.update_perf(old_model_parameters)
                self.update_drop_list(drop_count=self.drop_count)
            else:
                time.sleep(1)

    def process_packet_server_to_client(self, scapy_pkt):
        # Store IPs that are seen receiving big packets from server (global model)
        if scapy_pkt['IP'].dst not in self.clients_training and self.current_round > 0:
            if self.current_round < fado_args.drop_start or scapy_pkt['IP'].dst not in self.ips_lowest_losses:
                clients_training_lock.acquire()
                logger.info(f"{scapy_pkt['IP'].dst} was chosen")
                self.clients_training.append(scapy_pkt['IP'].dst)
                clients_training_lock.release()

        return scapy_pkt

    def process_packet_client_to_server(self, scapy_pkt):
        if self.current_round >= fado_args.drop_start:
            if scapy_pkt['IP'].src in self.ips_lowest_losses:
                return None

        return scapy_pkt

    def update_perf(self, model_parameters):
        logger.info(f"Updating performance")
        # Evaluate loss of the current model parameters with attacker test set
        self.local_model.set_parameters(model_parameters)
        current_loss, current_acc = self.local_model.evaluate(self.x_target_test, self.y_target_test)
        logger.info(f"Attacker model loss, accuracy - {current_loss}, {current_acc}")
        # Calculate loss improvement
        if self.last_loss is None:
            self.last_loss = current_loss
            self.clients_prev_round = []
            return

        perf_diff = current_loss - self.last_loss
        logger.info(f"Performance improvement - {-perf_diff}")
        self.last_loss = current_loss
        # Update the mean of improvements of every client that participated in this round
        for client_ip in self.clients_prev_round:
            if client_ip not in self.clients_improv_history:
                self.clients_improv_history[client_ip] = []
            self.clients_improv_history[client_ip].append(perf_diff)

        # Reset list of clients that participated in the round
        self.clients_prev_round = []

    def update_drop_list(self, drop_count):
        # Choose lowest losses -> bigger improvements
        all_changes = [self.clients_improv_history.get(str(ipaddress.IPv4Address('10.128.1.1') + client_id), np.inf)
                       for client_id in range(fado_args.number_clients)]
        all_means = [np.mean(change) for change in all_changes]
        self.ips_lowest_losses = [str(ipaddress.IPv4Address('10.128.1.1') + client_id) for client_id in
                                  np.argsort(all_means)[:drop_count]]
        interception = len(set(self.ips_lowest_losses) & self.target_clients)
        logger.info(
            f'IPs to drop - {self.ips_lowest_losses}. Number of clients that have target class - {interception}')
