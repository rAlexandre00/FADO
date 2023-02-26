import logging
import subprocess
import time

from warnings import filterwarnings

import numpy as np

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import SERVER_PORT

filterwarnings("ignore")
from netfilterqueue import NetfilterQueue
from scapy.all import *
from scapy.layers.inet import IP

QUEUE_NUMBER = 2

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'router'})


def process_packet(pkt):
    scapy_pkt = IP(pkt.get_payload())
    #scapy_pkt = FadoAttacker.get_instance().attack_network(scapy_pkt)
    print(scapy_pkt, flush=True)
    pkt.set_payload(raw(scapy_pkt))
    pkt.accept()


if __name__ == "__main__":
    time.sleep(100000)
    args = FADOArguments('/app/config/fado_config.yaml')

    # Set the seed for PRNGs to be equal to the trial index
    seed = args.random_seed
    np.random.seed(seed)
    random.seed(seed)

    # load_attack(args, 'network_attack_spec')
    # FadoAttacker.get_instance().init(args, 'network_attack_spec')

    # if FadoAttacker.get_instance().is_network_attack():
    # Sniff packets and for each packet send it to attack (apply filter first)
    nfqueue = NetfilterQueue()

    p = subprocess.call(['iptables', '-I', 'FORWARD', '-p', 'tcp', '--destination-port', SERVER_PORT, '-j', 'NFQUEUE',
                         '--queue-num', f'{QUEUE_NUMBER}'])

    # Bind to the same queue number (here 2)
    nfqueue.bind(QUEUE_NUMBER, process_packet)

    print('Starting network attack')
    # run (indefinitely)
    try:
        nfqueue.run()
    except KeyboardInterrupt:
        print('Quiting...')
    nfqueue.unbind()
