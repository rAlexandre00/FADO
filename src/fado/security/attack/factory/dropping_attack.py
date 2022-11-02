import logging

from fado.security.attack.attack_base import NetworkAttack

logger = logging.getLogger("fado")


class DroppingAttack(NetworkAttack):

    def __init__(self):
        super().__init__()

    def attack_network(self, packet):
        logger.info(packet)
