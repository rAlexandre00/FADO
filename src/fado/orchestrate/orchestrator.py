import pathlib
from distutils.dir_util import copy_tree
from pathlib import Path

import yaml
import shutil

from fado.arguments import AttackArguments
from fado.constants import *
from fado.crypto.hash_verifier import file_changed, write_file_hash

import logging

logger = logging.getLogger("fado")

__all__ = ['prepare_orchestrate']


def prepare_orchestrate(config_path, args, dev=False):
    """Creates the necessary files so that 'docker compose up' is possible
        1 - Generate image files
        2 - Generate docker-compose file
        3 - Create grpc_ipconfig file and fedml_config.yaml
        4 - Generate tls certificates
        5 - Split data for each client for train and test

        Parameters:
            config_path(str): Path for the yaml configuration file
            args(AttackArguments): FADO arguments
            dev: Identifies if development mode is enabled

    """
    os.makedirs(FADO_DIR, exist_ok=True)
    config_changed = file_changed(config_path, CONFIG_HASH)
    # pedreiro
    config_changed = True
    if not config_changed:
        logger.warning('Attack config has not changed. Data and configuration files will not change')
    else:
        write_file_hash(config_path, CONFIG_HASH)

    # if user omits args.model in fado_config.yaml we want it to be blank
    args.model = args.model if 'model' in args else ''

    if config_changed:
        logger.info("Creating docker files")
        # Generate image files
        generate_image_files(args.model)

        # Generate docker-compose file
        benign_ranks, malicious_ranks = generate_compose(args.dataset, args.benign_clients, args.malicious_clients,
                                                         DOCKER_COMPOSE_OUT, args.using_gpu)

        logger.info("Creating networking files")
        # Create script that will tell how the router should forward packets
        generate_router_files(args, benign_ranks, malicious_ranks, dev)
        # Create grpc_ipconfig file and fedml_config.yaml
        create_ipconfig(GRPC_CONFIG_OUT, args.benign_clients + args.malicious_clients)
        # Create fedml_config.yaml for server/benign client and for malicious client
        create_fedml_config(args)
        create_fedml_config(args, True)

        if "encrypt_comm" in args and args.encrypt_comm:
            logger.info("Creating TLS certificates")
            # Generate tls certificates (if defined in attacks args)
            create_certs()

        # Commenting this section as will not be part of the compose part
        #logger.info("Creating partitions for server and clients")
        # Split data for each client for train and test
        #split_data(args.dataset, ALL_DATA_FOLDER, PARTITION_DATA_FOLDER, args.benign_clients + args.malicious_clients)

        logger.info("Creating needed folders")
        os.makedirs(TENSORBOARD_DIRECTORY, exist_ok=True)
        os.makedirs(ATTACK_DIRECTORY, exist_ok=True)
        os.makedirs(DEFENSE_DIRECTORY, exist_ok=True)

        logger.info("Creating logs directory")
        os.makedirs(LOGS_DIRECTORY, exist_ok=True)

    if dev:
        logger.info("Generating dev files")
        generate_dev_files()


def generate_router_files(args, benign_ranks, malicious_ranks, dev=False):
    generate_router_nat(benign_ranks, malicious_ranks, ROUTER_IMAGE)
    generate_router_image(args, dev)


def generate_router_nat(benign_ranks, malicious_ranks, router_user_path):
    """Creates a script in the router that will create port forwarding rules and enable NAT
       This enables the communication between nodes in different overlay networks

        Parameters:
            benign_ranks: Ranks of benign clients
            malicious_ranks: Ranks of malicious clients
            client_user_path: Path of the docker image to be generated

    """
    lines: list[str] = []
    # Wait for the resolution of all nodes names
    lines.append('while [ -z "$fado_server" ]; '
                 'do fado_server=$(dig +short fado_server A); '
                 'done' + "\n")
    for rank in benign_ranks:
        lines.append(f'while [ -z "$fado_client_{rank}" ]; '
                     f'do fado_client_{rank}=$(dig +short fado_beg-client-{rank} A); '
                     f'done' + "\n")
    for rank in malicious_ranks:
        lines.append(f'while [ -z "$fado_client_{rank}" ]; '
                     f'do fado_client_{rank}=$(dig +short fado_mal-client-{rank} A); '
                     f'done' + "\n")
    # Create port forwarding for each node
    lines.append(
        'iptables -t nat -A PREROUTING -p tcp --dport 8890 -j DNAT --to-destination "$fado_server":8890' + "\n")
    for rank in benign_ranks + malicious_ranks:
        lines.append(f'iptables -t nat -A PREROUTING -p tcp --dport {8890 + rank} -j DNAT '
                     f'--to-destination "$fado_client_{rank}":{8890 + rank}' + "\n")
    # Enables NAT for eth0 and eth1
    lines.append(f'iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE' + "\n")
    lines.append(f'iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE')

    with open(os.path.join(router_user_path, "apply_nat.sh"), "w") as f:
        f.writelines(lines)


def generate_router_image(args, dev=False):
    path = Path(FADO_CONFIG_OUT)
    os.makedirs(path.parent.absolute(), exist_ok=True)
    with open(FADO_CONFIG_OUT, 'w') as f:
        yaml.dump(args.__dict__, f)


def generate_image_files(model):
    """Creates the docker folder that has fedml and router docker files

        Parameters:
            model_file(str): Path for a path that defines the model to train
            dev: Identifies if development mode is enabled
    :return:
    """
    client_user_path = FEDML_IMAGE
    router_user_path = ROUTER_IMAGE

    # Copies docker files in fado library to user space
    copy_tree(CLIENT_PATH, client_user_path)
    copy_tree(ROUTER_PATH, router_user_path)

    if os.path.exists(model):  # if model is a file, copy it
        shutil.copy2(model, os.path.join(client_user_path, 'get_model.py'))
    else:
        shutil.copy2('get_model.py', os.path.join(client_user_path, 'get_model.py'))


def generate_dev_files():
    # Copies docker files in fado library to user space
    copy_tree(CLIENT_PATH, FEDML_IMAGE)
    #copy_tree(ROUTER_PATH, ROUTER_IMAGE)

    # Router dev files
    requirements_base_path = os.path.join(ROUTER_PATH, 'dev_requirements.txt')
    requirements_user_path = os.path.join(ROUTER_IMAGE, 'requirements.txt')
    shutil.copy2(requirements_base_path, requirements_user_path)
    fado_path = os.path.join(ROUTER_IMAGE, 'fado')
    root_folder = str(pathlib.Path(__file__).parents[3])
    fado_folder = str(pathlib.Path(__file__).parents[1])
    copy_tree(fado_folder, os.path.join(fado_path, 'src', 'fado'))
    shutil.copy2(os.path.join(root_folder, 'setup.py'),
                 os.path.join(fado_path, 'setup.py'))

    # Client dev files
    client_user_path = FEDML_IMAGE
    fado_path = os.path.join(client_user_path, 'fado')
    fado_folder = str(pathlib.Path(__file__).parents[1])
    root_folder = str(pathlib.Path(__file__).parents[3])
    copy_tree(fado_folder, os.path.join(fado_path, 'src', 'fado'))
    shutil.copy2(os.path.join(root_folder, 'setup.py'),
                 os.path.join(fado_path, 'setup.py'))
    shutil.copy2(os.path.join(DEV_PATH, 'Dockerfile'),
                 os.path.join(client_user_path, 'Dockerfile'))
    shutil.copy2(os.path.join(DEV_PATH, 'requirements.txt'),
                 os.path.join(client_user_path, 'requirements.txt'))


def create_ipconfig(ipconfig_out, num_clients):
    """ Creates ipconfig file that tells FedML the IP of each node

        Parameters:
            ipconfig_out(str): Path for writing the ipconfig file
            num_clients: Number of clients
    """
    path = Path(ipconfig_out)
    os.makedirs(path.parent.absolute(), exist_ok=True)
    with open(ipconfig_out, 'w') as f:
        f.write('receiver_id,ip\n')
        for rank in range(num_clients + 1):
            f.write(f'{rank},fado_router\n')


def generate_compose(dataset, number_ben_clients, number_mal_clients, docker_compose_out, using_gpu=False):
    """
    Loads a default compose file and generates 'number_clients' of client services

        Parameters:
            number_ben_clients (int): Number of benign clients
            number_mal_clients (int): Number of malicious clients
            docker_compose_out (str): Path of the output for the docker compose

        Returns:
            Two lists of integers representing the benign_ranks and malicious_ranks
    """
    import random
    import copy
    from ipaddress import IPv4Address

    client_ranks = list(range(1, number_ben_clients + number_mal_clients + 1))
    # Shuffle clients randomly (a seed can be set)
    random.shuffle(client_ranks)

    benign_ranks = client_ranks[:number_ben_clients]
    malicious_ranks = client_ranks[number_ben_clients:]
    logger.info(f'Benign clients - {benign_ranks}')
    logger.info(f'Malicious clients - {malicious_ranks}')

    # Load the default docker compose
    docker_compose = load_base_compose(using_gpu)

    # Generate benign compose services
    base = docker_compose['services']['beg-client']
    for client_rank in benign_ranks:
        client_compose = copy.deepcopy(base)
        client_compose['container_name'] += f'-{client_rank}'
        client_compose['environment'] += [f'FEDML_RANK={client_rank}']
        client_compose['volumes'] += [f'{PARTITION_DATA_FOLDER}/{dataset}/user_{client_rank}:/app/data/']
        docker_compose['services'][f'beg-client-{client_rank}'] = client_compose
    docker_compose['services'].pop('beg-client')

    # Generate malicious compose services
    base = docker_compose['services']['mal-client']
    for client_rank in malicious_ranks:
        client_compose = copy.deepcopy(base)
        client_compose['container_name'] += f'-{client_rank}'
        client_compose['environment'] += [f'FEDML_RANK={client_rank}']
        client_compose['volumes'] += f'{PARTITION_DATA_FOLDER}/{dataset}/user_{client_rank}:/app/data/'
        docker_compose['services'][f'mal-client-{client_rank}'] = client_compose
    docker_compose['services'].pop('mal-client')

    # Customize volume for data in server
    docker_compose['services']['server']['volumes'] += [f'{PARTITION_DATA_FOLDER}/{dataset}/server:/app/data/']

    with open(docker_compose_out, 'w') as f:
        yaml.dump(docker_compose, f, sort_keys=False)

    return benign_ranks, malicious_ranks


def create_fedml_config(args, malicious=False):
    """ Generates fedml_config files for FedML nodes

    Parameters:
        args: Arguments of FADO
        malicious: if this fedml_config is malicious

    """
    file_path = os.path.abspath(os.path.realpath(FEDML_CONFIG_FILE_PATH))

    # Load base docker compose file
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    client_num = args.benign_clients + args.malicious_clients

    if malicious:
        fedml_config_out = FEDML_BEN_CONFIG_OUT
        # maybe throw an exception, what if 'client_attack_spec' is not defined?
        # TODO: user has to be alerted
        if hasattr(args, 'client_attack_spec'):
            config['attack_args'] = {}
            config['attack_args']['client_attack_spec'] = args.client_attack_spec
    else:
        if hasattr(args, 'defense_spec'):
            config['defense_args'] = {}
            config['defense_args']['defense_spec'] = args.defense_spec
        fedml_config_out = FEDML_MAL_CONFIG_OUT

    config['common_args']['random_seed'] = args.random_seed
    config['train_args']['client_num_in_total'] = client_num
    config['train_args']['client_num_per_round'] = args.clients_per_round
    config['train_args']['comm_round'] = args.rounds
    config['train_args']['epochs'] = args.epochs
    config['train_args']['batch_size'] = args.batch_size
    config['device_args']['worker_num'] = client_num
    config['data_args']['dataset'] = args.dataset
    config['model_args']['model'] = args.model
    if "encrypt_comm" in args:
        config['comm_args']['encrypt'] = args.encrypt_comm
    with open(fedml_config_out, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


def create_certs():
    """Generates ca_certs and node_certs that are signed with the ca key
    """
    # os.makedirs(CERTS_OUT, exist_ok=True)
    # generate_self_signed_certs(out_key_file=os.path.join(certs_path, 'ca-key.pem'),
    #                            out_cert_file=os.path.join(certs_path, 'ca-cert.pem'))
    # generate_certs(out_key_file=os.path.join(certs_path, 'server-key.pem'),
    #                out_cert_file=os.path.join(certs_path, 'server-cert.pem'),
    #                ca_key_file=os.path.join(certs_path, 'ca-key.pem'),
    #                ca_cert_file=os.path.join(certs_path, 'ca-cert.pem'))
    """ Certs are pre-generated to simplify multi node setups """

    copy_tree(CERTS_PATH, CERTS_OUT)


def load_base_compose(using_gpu=False):
    """
    Reads four files in order to create a default docker-compose
    The files are:
    - A base file which has the outer structure of the compose file and specifies the path of the other three files
    - A server file that specifies the part of the server service
    - A benign client file that specifies the part of each of the benign client's services
    - A malicious client file that specifies the part of each of the malicious client's services

        Returns:
            docker_compose (dict): Specifies the docker compose default yaml
    """
    import os
    from fado.constants import GENERAL_COMPOSE_FILE_PATH
    dir_path = os.path.dirname(os.path.realpath(GENERAL_COMPOSE_FILE_PATH))

    # Load base docker compose file
    with open(GENERAL_COMPOSE_FILE_PATH, 'r') as file:
        docker_compose = yaml.load(file, Loader=yaml.FullLoader)

    """
    deploy:
        resources:
            reservations:
            devices:
            - driver: nvidia
                capabilities: [gpu]
    """
    # Put inside the docker compose file the client and server base files
    for service in ['server', 'beg-client', 'mal-client', 'router']:
        with open(dir_path + os.path.sep + docker_compose['services'][service]['compose-file'], 'r') as file:
            compose = yaml.load(file, Loader=yaml.FullLoader)
            docker_compose['services'][service].pop('compose-file')
            docker_compose['services'][service] = compose      

    if using_gpu:
        with open(dir_path + os.path.sep + 'gpu_compose.yaml', 'r') as file:
            gpu_compose = yaml.load(file, Loader=yaml.FullLoader)
            docker_compose['services']['beg-client']['deploy'] = gpu_compose['deploy']
            docker_compose['services']['mal-client']['deploy'] = gpu_compose['deploy']
            docker_compose['services']['server']['deploy']['resources'] = gpu_compose['deploy']['resources']
    return docker_compose
