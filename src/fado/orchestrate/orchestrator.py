import os
from distutils.dir_util import copy_tree

import yaml
import shutil

from fado.arguments import AttackArguments
from fado.data import split_data
from fado.crypto import generate_self_signed_certs, generate_certs
from fado.constants import FEDML_CONFIG_FILE_PATH

import logging


def prepare_orchestrate(config_path, dev=False):
    args = AttackArguments(config_path)

    # Generate image files
    generate_image_files(args.model_file, dev)

    # Generate docker-compose file
    client_ranks = generate_compose(args.benign_clients, args.malicious_clients, args.docker_compose_out)

    # Create grpc_ipconfig file and fedml_config.yaml
    create_ipconfig(args.grpc_ipconfig_out, client_ranks, args.benign_clients)

    # Create fedml_config.yaml for server/benign client and for malicious client
    create_fedml_config(args)
    create_fedml_config(args, True)

    # Generate tls certificates (if defined in attacks args)
    create_certs()

    # Split data for each client for train and test
    split_data(args.all_data_folder, args.partition_data_folder, args.benign_clients + args.malicious_clients)


def generate_image_files(model_file, dev=False):
    from fado.constants import CLIENT_PATH, ROUTER_PATH, MAL_CLIENT_PATH
    docker_path = os.path.join('.', 'docker')
    client_path = os.path.join(docker_path, 'client')
    router_path = os.path.join(docker_path, 'router')
    if os.path.exists(docker_path):
        logging.warning('Docker folder already exist. Images files will not be replaced')
        return
    copy_tree(CLIENT_PATH, client_path)
    copy_tree(ROUTER_PATH, router_path)
    shutil.copy2(model_file, os.path.join(client_path, 'get_model.py'))
    if dev:
        import pathlib
        import fado.docker.dev
        fado_path = os.path.join(client_path, 'fado')
        fado_folder = str(pathlib.Path(__file__).parents[1])
        root_folder = str(pathlib.Path(__file__).parents[3])
        copy_tree(fado_folder, os.path.join(fado_path, 'src', 'fado'))
        shutil.copy2(os.path.join(root_folder, 'setup.py'),
                        os.path.join(fado_path, 'setup.py'))
        shutil.copy2(os.path.join(os.path.dirname(fado.docker.dev.__file__), 'Dockerfile'),
                        os.path.join(client_path, 'Dockerfile'))
        shutil.copy2(os.path.join(os.path.dirname(fado.docker.dev.__file__), 'requirements.txt'),
                        os.path.join(client_path, 'requirements.txt'))


def create_ipconfig(ipconfig_out, client_ranks, number_ben_clients):
    from ipaddress import IPv4Address
    if os.path.exists(ipconfig_out):
        logging.warning('ipconfig already exists. ipconfig will not be replaced')
        return
    client_base_address = IPv4Address("172.10.1.0")
    with open(ipconfig_out, 'w') as f:
        f.write('receiver_id,ip\n')
        f.write(f'0,"172.20.1.0"\n')
        for rank in client_ranks:
            f.write(f'{rank},{client_base_address + rank}\n')


def generate_compose(number_ben_clients, number_mal_clients, docker_compose_out):
    """
    Loads a default compose file and generates 'number_clients' of client services

        Parameters:
            number_ben_clients (int): Number of benign clients
            number_mal_clients (int): Number of malicious clients
            docker_compose_out (str): Path of the output for the docker compose

        Returns:
            docker_compose (dict): Specifies the docker compose yaml
    """
    import random
    import copy
    from ipaddress import IPv4Address

    client_ranks = list(range(1, number_ben_clients + number_mal_clients + 1))
    # TODO: Read from config file seed value
    # random.shuffle(client_ranks)

    if os.path.exists(docker_compose_out):
        logging.warning('Compose file already exists. Compose file will not be replaced')
        return client_ranks

    # Load the default docker compose
    docker_compose = load_base_compose()

    # Generate benign compose services
    base = docker_compose['services']['fedml-client-benign']
    for client_rank in client_ranks[number_ben_clients:]:
        client_compose = copy.deepcopy(base)
        client_compose['container_name'] += f'-{client_rank}'
        client_compose['environment'] += [f'FEDML_RANK={client_rank}']
        client_compose['volumes'] += [f'./data/partitions/user_{client_rank}:/app/data/']
        client_ipv4_address = IPv4Address(client_compose['networks']['clients_network']['ipv4_address']) + client_rank
        client_compose['networks']['clients_network']['ipv4_address'] = str(client_ipv4_address)
        docker_compose['services'][f'fedml-beg-client-{client_rank}'] = client_compose
    docker_compose['services'].pop('fedml-client-benign')

    # Generate malicious compose services
    base = docker_compose['services']['fedml-client-malicious']
    for client_rank in client_ranks[:number_ben_clients]:
        client_compose = copy.deepcopy(base)
        client_compose['container_name'] += f'-{client_rank}'
        client_compose['environment'] += [f'FEDML_RANK={client_rank}']
        client_compose['volumes'] += [f'./data/partitions/user_{client_rank}:/app/data/']
        client_ipv4_address = IPv4Address(client_compose['networks']['clients_network']['ipv4_address']) + client_rank
        client_compose['networks']['clients_network']['ipv4_address'] = str(client_ipv4_address)
        docker_compose['services'][f'fedml-mal-client-{client_rank}'] = client_compose
    docker_compose['services'].pop('fedml-client-malicious')

    with open(docker_compose_out, 'w') as f:
        yaml.dump(docker_compose, f, sort_keys=False)

    return client_ranks


def create_fedml_config(args, malicious=False):
    file_path = os.path.abspath(os.path.realpath(FEDML_CONFIG_FILE_PATH))

    # Load base docker compose file
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    client_num = args.benign_clients + args.malicious_clients

    if malicious:
        fedml_config_out = args.fedml_config_out_malicious
        if os.path.exists(fedml_config_out):
            logging.warning('Malicious fedml_config already exists. Malicious fedml_config will not be replaced')
            return
        # maybe throw an exception, what if 'attack_spec' is not defined?
        # TODO: user has to be alerted
        config['attack_args'] = {}
        config['attack_args']['attack_spec'] = args.attack_spec
    else:
        fedml_config_out = args.fedml_config_out
        if os.path.exists(fedml_config_out):
            logging.warning('Benign fedml_config already exists. Benign fedml_config will not be replaced')
            return

    config['train_args']['client_num_in_total'] = client_num
    with open(fedml_config_out, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


def create_certs():
    certs_path = os.path.join('.', 'certs')
    if os.path.exists(certs_path):
        logging.warning('Certs folder already exists. Certs will not be replaced')
        return
    os.makedirs(certs_path, exist_ok=True)
    generate_self_signed_certs(out_key_file=os.path.join(certs_path, 'ca-key.pem'),
                               out_cert_file=os.path.join(certs_path, 'ca-cert.pem'))
    generate_certs(out_key_file=os.path.join(certs_path, 'server-key.pem'),
                   out_cert_file=os.path.join(certs_path, 'server-cert.pem'),
                   ca_key_file=os.path.join(certs_path, 'ca-key.pem'),
                   ca_cert_file=os.path.join(certs_path, 'ca-cert.pem'))


def load_base_compose():
    """
    Reads three files in order to create a default docker-compose
    The files are:
    - A base file which has the outer structure of the compose file and specifies the path of the other three files
    - A server file that specifies the part of the server service
    - A server file that specifies the part of each of the benign client's services
    - A server file that specifies the part of each of the malicious client's services

        Parameters:
            base_file_path (str): Specifies the path of the base file

        Returns:
            docker_compose (dict): Specifies the docker compose default yaml
    """
    import os
    from fado.constants import GENERAL_COMPOSE_FILE_PATH
    dir_path = os.path.dirname(os.path.realpath(GENERAL_COMPOSE_FILE_PATH))

    # Load base docker compose file
    with open(GENERAL_COMPOSE_FILE_PATH, 'r') as file:
        docker_compose = yaml.load(file, Loader=yaml.FullLoader)

    # Put inside the docker compose file the client and server base files
    for service in ['fedml-server', 'fedml-client-benign', 'fedml-client-malicious', 'fado-router']:
        with open(dir_path + os.path.sep + docker_compose['services'][service]['compose-file'], 'r') as file:
            compose = yaml.load(file, Loader=yaml.FullLoader)
            docker_compose['services'][service].pop('compose-file')
            docker_compose['services'][service] = compose
    return docker_compose
