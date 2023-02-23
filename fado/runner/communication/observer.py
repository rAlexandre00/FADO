from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def receive_message(self, message) -> None:
        pass
