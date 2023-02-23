import random
from typing import List

from fado.runner.fl.select.base_participant_selector import ParticipantSelector


class RandomParticipantSelector(ParticipantSelector):

    def get_participants(self, available_clients: List, num: int):
        return random.sample(available_clients, num)
