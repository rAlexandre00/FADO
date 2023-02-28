import ipaddress
import os
import subprocess
import sys
import time
import socket

from fado.cli.arguments.arguments import FADOArguments

args = FADOArguments(os.getenv("FADO_CONFIG_PATH", default="/app/config/fado_config.yaml"))


def create_interface(ip, mask):
    subprocess.run(['ip', 'link', 'add', 'link', 'eth0', 'name', ip, 'type', 'ipvlan', 'mode', 'l2'])
    subprocess.run(['ip', 'addr', 'add', 'dev', ip, f'{ip}/{mask}'])
    subprocess.run(['ip', 'link', 'set', 'dev', ip, 'up'])


def create_interfaces(n):
    base_ip = ipaddress.ip_address('10.128.1.0')
    for i in range(n+1):
        create_interface(str(base_ip + i), 9)
        subprocess.run(
            ['ip', 'route', 'add', 'table', f'{2 + i}', 'to', 'default', 'via', '10.128.0.1', 'dev', str(base_ip + i)])
        subprocess.run(['ip', 'rule', 'add', 'from', f'{str(base_ip + i)}/32', 'table', f'{2 + i}', 'priority', '10'])


def config():
    create_interfaces(args.number_clients)


config()
sys.stdout.flush()
