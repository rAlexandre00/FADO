import subprocess

from netfilterqueue import NetfilterQueue
from scapy.all import *
from scapy.layers.inet import IP

from fado.arguments import AttackArguments
from fado.constants import FADO_CONFIG_OUT
from fado.logging.prints import HiddenPrints
from fado.security.attack import FadoAttacker
from fado.security.utils import load_attack

QUEUE_NUMBER = 2


def process_packet(pkt):
    pkt = IP(pkt.get_payload())
    pkt = FadoAttacker.get_instance().attack_network(pkt)
    pkt.accept()


if __name__ == "__main__":
    args = AttackArguments(FADO_CONFIG_OUT)

    load_attack(args, 'network_attack_spec')
    FadoAttacker.get_instance().init(args, 'network_attack_spec')

    if FadoAttacker.get_instance().is_network_attack():
        # Sniff packets and for each packet send it to attack (apply filter first)
        nfqueue = NetfilterQueue()

        FadoAttacker.get_instance().get_network_filter()
        p = subprocess.run(f"iptables -I INPUT -p tcp --destination-port 8890 -j NFQUEUE --queue-num {QUEUE_NUMBER}")

        # Bind to the same queue number (here 2)
        nfqueue.bind(QUEUE_NUMBER, process_packet)

        # run (indefinitely)
        try:
            nfqueue.run()
        except KeyboardInterrupt:
            print('Quiting...')
        nfqueue.unbind()
