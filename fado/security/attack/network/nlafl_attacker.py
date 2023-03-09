import heapq
import ipaddress
import logging
import os
import pickle
import socket
import struct
import time

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_PUB_PORT
from fado.runner.communication.message import Message
from fado.runner.communication.sockets.utils import recvall

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'router'})
fado_args = FADOArguments("/app/config/fado_config.yaml")

BASE_IP = ipaddress.ip_address('10.128.1.0')


def get_model_parameters():
    # Connect
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((os.getenv('SERVER_IP'), SERVER_PUB_PORT))

    # Send model request
    message = Message(sender_id=-1, receiver_id=-1, type=Message.MSG_TYPE_GET_MODEL)
    message_encoded = pickle.dumps(message)
    s.sendall(struct.pack('>I', len(message_encoded)))
    s.sendall(message_encoded)

    # Receive model parameters
    message_size = struct.unpack('>I', recvall(s, 4))[0]
    message_encoded = recvall(s, message_size)
    message = pickle.loads(message_encoded)
    return message.get(Message.MSG_ARG_KEY_MODEL_PARAMS)


class NLAFLAttacker:

    def __init__(self, model, x_target_test, y_target_test):
        """ Called when receives message

        :param model (FADOModule):
        :return:
        """
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

    def process_packet_server_to_client(self, scapy_pkt):
        # Store IPs that are seen receiving big packets from server (global model)
        if scapy_pkt['IP'].dst not in self.clients_training:
            self.clients_training.append(scapy_pkt['IP'].dst)
            if self.server_is_aggregating:
                logger.info(f"Round {self.current_round} start detected")
                self.server_is_aggregating = False
                if self.current_round > 0:
                    # Server started new round - estimate contributions for the previous round
                    self.update_perf()
                    self.update_drop_list(drop_count=fado_args.drop_count)
            self.clients_prev_round.append(scapy_pkt['IP'].dst)

        return scapy_pkt

    def process_packet_client_to_server(self, scapy_pkt):
        if self.current_round >= fado_args.drop_start:
            if scapy_pkt['IP'].src in self.ips_lowest_losses:
                if scapy_pkt['IP'].src in self.clients_training:
                    self.client_ended_training(scapy_pkt['IP'].src)
                return None

        if scapy_pkt['IP'].src in self.clients_training:
            self.client_ended_training(scapy_pkt['IP'].src)

        return scapy_pkt

    def client_ended_training(self, ip):
        self.clients_training.remove(ip)
        # If list of clients training is empty then all clients sent their local models and server is aggregating
        if len(self.clients_training) == 0:
            logger.info(f"Round {self.current_round} finish detected")
            self.current_round += 1
            self.server_is_aggregating = True

    def update_perf(self):
        # Evaluate loss of the current model parameters with attacker test set
        self.local_model.set_parameters(get_model_parameters())
        current_loss, _ = self.local_model.evaluate(self.x_target_test, self.y_target_test)

        # Calculate loss improvement
        if self.last_loss is None:
            self.last_loss = current_loss
            self.clients_prev_round = []
            return

        perf_diff = current_loss - self.last_loss
        self.last_loss = current_loss
        # Update the mean of improvements of every client that participated in this round
        for client_ip in self.clients_prev_round:
            if client_ip not in self.clients_improv_history:
                self.clients_improv_history[client_ip] = (0, 0)
            sum_perf, count_perf = self.clients_improv_history[client_ip]
            self.clients_improv_history[client_ip] = (
                (sum_perf * count_perf + perf_diff) / (count_perf + 1), count_perf + 1)

        # Reset list of clients that participated in the round
        self.clients_prev_round = []

    def update_drop_list(self, drop_count):
        # Choose lowest losses -> bigger improvements
        clients_lowest_losses = heapq.nsmallest(drop_count, self.clients_improv_history.items(), key=lambda x: x[1][0])
        self.ips_lowest_losses = [x[0] for x in clients_lowest_losses]
        interception = len(set(self.ips_lowest_losses) & self.target_clients)
        logger.info(
            f'IPs to drop - {self.ips_lowest_losses}. Number of clients that have target class - {interception}')
