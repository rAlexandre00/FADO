import subprocess

from warnings import filterwarnings

import numpy as np

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_PORT
from fado.runner.data.load.attacker_data_loader import AttackerDataLoader
from fado.runner.ml.model.module_manager import ModelManager
from fado.security.attack.network.network_attacker import NetworkAttacker

filterwarnings("ignore")
from netfilterqueue import NetfilterQueue
from scapy.all import *
from scapy.layers.inet import IP

QUEUE_NUMBER_CLIENT_TO_SERVER = 2
QUEUE_NUMBER_SERVER_TO_CLIENT = 3

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'router'})

network_attacker = None


def process_packet_client_to_server(pkt):
    scapy_pkt = IP(pkt.get_payload())
    # Ignore if client node is checking if server is alive
    if scapy_pkt.getlayer("IP").src != '10.128.1.0':
        scapy_pkt = network_attacker.process_packet_client_to_server(scapy_pkt)
        pkt.set_payload(raw(scapy_pkt))
    pkt.accept()


def process_packet_server_to_client(pkt):
    scapy_pkt = IP(pkt.get_payload())
    # Ignore if client node is checking if server is alive
    if scapy_pkt.getlayer("IP").dst != '10.128.1.0':
        scapy_pkt = network_attacker.process_packet_server_to_client(scapy_pkt)
        pkt.set_payload(raw(scapy_pkt))
    pkt.accept()


if __name__ == "__main__":
    args = FADOArguments('/app/config/fado_config.yaml')

    data_loader = AttackerDataLoader('/app/data')
    dataset = data_loader.read_data()

    network_attacker = NetworkAttacker(ModelManager.get_model(), dataset.test_data['x'], dataset.test_data['y'])

    # Set the seed for PRNGs to be equal to the trial index
    seed = args.random_seed
    np.random.seed(seed)
    random.seed(seed)

    # load_attack(args, 'network_attack_spec')
    # FadoAttacker.get_instance().init(args, 'network_attack_spec')

    # if FadoAttacker.get_instance().is_network_attack():
    # Sniff packets and for each packet send it to attack (apply filter first)
    nfqueue_client_to_server = NetfilterQueue()
    nfqueue_server_to_client = NetfilterQueue()

    p = subprocess.call(
        ['iptables', '-I', 'FORWARD', '-p', 'tcp', '--destination-port', str(SERVER_PORT),
         '-m', 'length', '!', '--length', '0:500',  # Ignore handshake and control packets
         '-j', 'NFQUEUE', '--queue-num', f'{QUEUE_NUMBER_CLIENT_TO_SERVER}'])
    p = subprocess.call(
        ['iptables', '-I', 'FORWARD', '-p', 'tcp', '--source-port', str(SERVER_PORT),
         '-m', 'length', '!', '--length', '0:500',  # Ignore handshake and control packets
         '-j', 'NFQUEUE', '--queue-num', f'{QUEUE_NUMBER_SERVER_TO_CLIENT}'])

    # Bind to the same queue number (here 2)
    nfqueue_client_to_server.bind(QUEUE_NUMBER_CLIENT_TO_SERVER, process_packet_client_to_server)
    # Bind to the same queue number (here 3)
    nfqueue_server_to_client.bind(QUEUE_NUMBER_SERVER_TO_CLIENT, process_packet_server_to_client)

    print('Starting network attack')
    # run (indefinitely)
    try:
        Thread(target=nfqueue_client_to_server.run, daemon=True).start()
        Thread(target=nfqueue_server_to_client.run, daemon=True).start()
        time.sleep(999999999)
    except KeyboardInterrupt:
        print('Quiting...')
    finally:
        nfqueue_client_to_server.unbind()
        nfqueue_server_to_client.unbind()
