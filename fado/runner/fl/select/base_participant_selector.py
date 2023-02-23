from abc import abstractmethod


class ParticipantSelector(object):

    def __init__(self):
        pass

    @abstractmethod
    def get_participants(self, available_clients, num):
        pass
