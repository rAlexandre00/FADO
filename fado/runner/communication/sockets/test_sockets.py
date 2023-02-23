import os

from fado.runner.communication.sockets.server_com_manager import SocketCommunicationManager

if __name__ == '__main__':
    SocketCommunicationManager(int(os.getenv("FADO_ID")))
