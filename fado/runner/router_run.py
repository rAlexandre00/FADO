import subprocess

from warnings import filterwarnings
import multiprocessing

import numpy as np

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_PORT
from fado.runner.data.load.attacker_data_loader import AttackerDataLoader
from fado.runner.ml.model.module_manager import ModelManager
from fado.security.attack.network.network_attack_manager import NetworkAttackerManager

filterwarnings("ignore")
from netfilterqueue import NetfilterQueue
from scapy.all import *
from scapy.layers.inet import IP

QUEUE_NUMBER_CLIENT_TO_SERVER = 2
QUEUE_NUMBER_SERVER_TO_CLIENT = 40

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'router'})

network_attacker = None


def process_packet_client_to_server(pkt):
    scapy_pkt = IP(pkt.get_payload())
    # Ignore if client node is checking if server is alive
    if network_attacker is not None:
        scapy_pkt = network_attacker.process_packet_client_to_server(scapy_pkt)
    if scapy_pkt is None:
        pkt.drop()
    else:
        pkt.set_payload(raw(scapy_pkt))
        pkt.accept()


def process_packet_server_to_client(pkt):
    scapy_pkt = IP(pkt.get_payload())
    # Ignore if client node is checking if server is alive
    if network_attacker is not None:
        scapy_pkt = network_attacker.process_packet_server_to_client(scapy_pkt)
    if scapy_pkt is None:
        pkt.drop()
    else:
        pkt.set_payload(raw(scapy_pkt))
        pkt.accept()


if __name__ == "__main__":
    args = FADOArguments('/app/config/fado_config.yaml')

    data_loader = AttackerDataLoader('/app/data')
    dataset = data_loader.read_data()

    network_attacker = NetworkAttackerManager.get_attacker(ModelManager.get_model(), dataset.test_data['x'], dataset.test_data['y'])

    # Set the seed for PRNGs to be equal to the trial index
    seed = args.random_seed
    np.random.seed(seed)
    random.seed(seed)

    # load_attack(args, 'network_attack_spec')
    # FadoAttacker.get_instance().init(args, 'network_attack_spec')

    # if FadoAttacker.get_instance().is_network_attack():
    # Sniff packets and for each packet send it to attack (apply filter first)
    queue_numer = 2

    client_to_server_queues = [NetfilterQueue() for _ in range(queue_numer)]
    server_to_client_queues = [NetfilterQueue() for _ in range(queue_numer)]

    p = subprocess.call(
        ['iptables', '-I', 'FORWARD', '-p', 'tcp', '--destination-port', str(SERVER_PORT),
         '-m', 'length', '!', '--length', '0:500',  # Ignore handshake and control packets
         '-j', 'NFQUEUE', '--queue-balance', f'{QUEUE_NUMBER_CLIENT_TO_SERVER}:{QUEUE_NUMBER_CLIENT_TO_SERVER+queue_numer-1}'])
    p = subprocess.call(
        ['iptables', '-I', 'FORWARD', '-p', 'tcp', '--source-port', str(SERVER_PORT),
         '-m', 'length', '!', '--length', '0:500',  # Ignore handshake and control packets
         '-j', 'NFQUEUE', '--queue-balance', f'{QUEUE_NUMBER_SERVER_TO_CLIENT}:{QUEUE_NUMBER_SERVER_TO_CLIENT+queue_numer-1}'])

    # Bind to the same queue number
    for queue_numer_offset, nfqueue_client_to_server in enumerate(client_to_server_queues):
        nfqueue_client_to_server.bind(QUEUE_NUMBER_CLIENT_TO_SERVER+queue_numer_offset, process_packet_client_to_server)
        threading.Thread(target=nfqueue_client_to_server.run, daemon=True).start()
    # Bind to the same queue number
    for queue_numer_offset, nfqueue_server_to_client in enumerate(server_to_client_queues):
        nfqueue_server_to_client.bind(QUEUE_NUMBER_SERVER_TO_CLIENT+queue_numer_offset, process_packet_server_to_client)
        threading.Thread(target=nfqueue_server_to_client.run, daemon=True).start()

    try:
        time.sleep(999999999)
    except KeyboardInterrupt:
        print('Quiting...')
