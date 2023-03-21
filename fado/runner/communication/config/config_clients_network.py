import ipaddress
import logging
import os
import subprocess
import sys
import time
import socket

from fado.cli.arguments.arguments import FADOArguments

args = FADOArguments(os.getenv("FADO_CONFIG_PATH", default="/app/config/fado_config.yaml"))
logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': 'clients'})


def create_interface(ip, mask):
    subprocess.run(['ip', 'link', 'add', 'link', 'eth0', 'name', ip, 'type', 'ipvlan', 'mode', 'l2'])
    subprocess.run(['ip', 'addr', 'add', 'dev', ip, f'{ip}/{mask}'])
    subprocess.run(['ip', 'link', 'set', 'dev', ip, 'up'])


def create_interfaces(n):
    base_ip = ipaddress.ip_address('10.128.1.0')
    created_interfaces = 0
    for i in range(n + 3):
        ip = str(base_ip + created_interfaces)
        rule_number = 2+i
        if 2 + i == 254:
            continue
        create_interface(str(base_ip + created_interfaces), 9)
        subprocess.run(
            ['ip', 'route', 'add', 'table', f'{rule_number}', 'to', 'default', 'via', '10.128.0.1', 'dev', ip])
        subprocess.run(['ip', 'rule', 'add', 'from', f'{ip}/32', 'table', f'{rule_number}', 'priority', '10'])
        created_interfaces += 1
        if created_interfaces == n + 1:
            break


def config():
    create_interfaces(args.number_clients)


config()
sys.stdout.flush()
