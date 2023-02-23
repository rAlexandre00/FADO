import subprocess
import sys


def create_interface(ip, mask):
    subprocess.run(['ip', 'link', 'add', 'link', 'eth0', 'name', ip, 'type', 'ipvlan', 'mode', 'l2'])
    subprocess.run(['ip', 'addr', 'add', 'dev', ip, f'{ip}/{mask}'])
    subprocess.run(['ip', 'link', 'set', 'dev', ip, 'up'])


def config():
    create_interface('10.0.0.1', 24)
    create_interface('10.128.0.1', 9)


config()
sys.stdout.flush()
