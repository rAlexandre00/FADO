import subprocess
import sys


def create_interface(ip, mask, clients_network, gateway):
    subprocess.run(['ip', 'link', 'add', 'link', 'eth0', 'name', ip, 'type', 'ipvlan', 'mode', 'l2'])
    subprocess.run(['ip', 'addr', 'add', 'dev', ip, f'{ip}/{mask}'])
    subprocess.run(['ip', 'link', 'set', 'dev', ip, 'up'])
    subprocess.run(['ip', 'route', 'add', clients_network, 'dev', ip, 'via', gateway])

def config():
    create_interface('10.0.0.2', 24, '10.128.0.0/9', '10.0.0.1')

config()
sys.stdout.flush()
