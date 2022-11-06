import logging
from statistics import mean
from time import sleep
from threading import Thread

import requests
import torch
from sklearn.metrics import mean_squared_error

from fado.security.attack.attack_base import NetworkAttack

logger = logging.getLogger("fado")
global_model = None

def _get_global_model():
    global global_model
    while True:
        try:
            global_model = requests.get('http://fado_server:8889/global_model')
            # logger.info(global_model.text)
        except requests.exceptions.ConnectionError as e:
            pass
            # logger.info(e)
        sleep(1)


class DroppingAttack(NetworkAttack):

    def __init__(self, args):
        super().__init__()
        self.clients_loss_difference = {}
        self.last_round_loss = 0
        #self.dataset = # Get from args
        self.rounds_passed = 0
        self.args = args
        t1 = Thread(target=_get_global_model)
        t1.start()

    def attack_network(self, packet):
        # logger.info(packet.summary())
        # TODO: Get required information
        global global_model
        # Test if global_parameters changed
        new_round = False
        if new_round:
            # Append to participants clients communicating in that round
            participants = []
            self.rounds_passed += 1
            participants = []
            self._update_client_loss(participants, global_model)
        if self.rounds_passed == self.args.rounds:
            self.rounds_passed = 0
            largest_value_clients = self._get_largest_values()
            logger.info(f'Dropping traffic for clients: {largest_value_clients}')
        return packet

    def _update_client_loss(self, participants, global_model):
        loss_difference = self._calculate_loss_difference(global_model)
        for client in participants:
            self.clients_loss_difference[client].append([loss_difference])

    def _calculate_loss_difference(self, current_model):
        this_round_loss = self._calculate_loss(current_model, torch.nn.CrossEntropyLoss())
        loss_difference = self.last_round_loss - this_round_loss
        self.last_round_loss = this_round_loss
        return loss_difference

    def _calculate_loss(self, model, criterion):
        # TODO: Put the right way to get dataset x and y
        prediction = model(self.dataset.x)
        return criterion(prediction, self.dataset.y)

    def _get_largest_values(self):
        for client in self.clients_loss_difference.keys():
            self.clients_loss_difference[client] = mean(self.clients_loss_difference[client])
        # Chooses clients with biggest values
        largest_value_clients = sorted(self.clients_loss_difference.items(), key=lambda x: x[1])[:self.args.clients_drop]
        return largest_value_clients