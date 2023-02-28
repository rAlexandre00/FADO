import socket
import subprocess
import sys


def create_interface(ip, mask, interface):
    subprocess.run(['ip', 'link', 'add', 'link', interface, 'name', ip, 'type', 'ipvlan', 'mode', 'l2'])
    subprocess.run(['ip', 'addr', 'add', 'dev', ip, f'{ip}/{mask}'])
    subprocess.run(['ip', 'link', 'set', 'dev', ip, 'up'])


def config():
    create_interface('10.0.0.1', 24, 'eth0')
    create_interface('10.128.0.1', 9, 'eth1')


config()
sys.stdout.flush()
