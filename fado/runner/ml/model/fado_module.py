from abc import abstractmethod


class FADOModule(object):

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, new_weights):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def evaluate(self, x, y):
        pass
