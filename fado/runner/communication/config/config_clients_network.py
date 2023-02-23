import ipaddress
import os
import subprocess
import sys
import time

from fado.cli.arguments.arguments import FADOArguments

args = FADOArguments(os.getenv("FADO_CONFIG_PATH", default="/app/config/fado_config.yaml"))

def create_interface(ip, mask):
    subprocess.run(['ip', 'link', 'add', 'link', 'eth0', 'name', ip, 'type', 'ipvlan', 'mode', 'l2'])
    subprocess.run(['ip', 'addr', 'add', 'dev', ip, f'{ip}/{mask}'])
    subprocess.run(['ip', 'link', 'set', 'dev', ip, 'up'])

def create_interfaces(n):
    base_ip = ipaddress.ip_address('10.128.0.2')
    for i in range(n):
        create_interface(str(base_ip + i), 9)
    # TODO: DELETE route in favor of socket options
    subprocess.run(['ip', 'route', 'add', '10.0.0.2/32', 'dev', str(base_ip), 'via', str(base_ip - 1)])


def config():
    create_interfaces(args.number_clients)

config()
sys.stdout.flush()
