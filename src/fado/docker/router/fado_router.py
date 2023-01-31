import logging

from warnings import filterwarnings
filterwarnings("ignore")
from netfilterqueue import NetfilterQueue
from scapy.all import *
from scapy.layers.inet import IP

from fado.arguments import AttackArguments
from fado.security.attack import FadoAttacker
from fado.security.utils import load_attack

QUEUE_NUMBER = 2

logger = logging.getLogger("fado")


def process_packet(pkt):
    scapy_pkt = IP(pkt.get_payload())
    print(scapy_pkt)
    scapy_pkt = FadoAttacker.get_instance().attack_network(scapy_pkt)
    pkt.set_payload(raw(scapy_pkt))
    pkt.accept()


if __name__ == "__main__":
    args = AttackArguments('config/fado_config.yaml')

    load_attack(args, 'network_attack_spec')
    FadoAttacker.get_instance().init(args, 'network_attack_spec')

    if FadoAttacker.get_instance().is_network_attack():
        # Sniff packets and for each packet send it to attack (apply filter first)
        nfqueue = NetfilterQueue()

        p = subprocess.call(['iptables', '-I', 'FORWARD', '-p', 'tcp', '--destination-port', '8890', '-j', 'NFQUEUE',
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
