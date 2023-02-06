import ipaddress
import os
import pathlib
import random
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

        benign_ranks, malicious_ranks = generate_client_ranks(args.benign_clients, args.malicious_clients)

        # Generate docker-compose file
        generate_compose(args.dataset, DOCKER_COMPOSE_OUT, args.benign_clients + args.malicious_clients, args.using_gpu)

        logger.info("Creating networking files")
        # Create script that will tell how the router should forward packets
        generate_router_files(args, benign_ranks, malicious_ranks, dev)
        # Create grpc_ipconfig file and fedml_config.yaml
        for rank in [0]+benign_ranks+malicious_ranks:
            create_ipconfig(CONFIG_OUT, rank, args.benign_clients + args.malicious_clients)
        # Create fedml_config.yaml for server/benign client and for malicious client
        create_fedml_config(args, 0)
        for rank in benign_ranks:
            create_fedml_config(args, rank)
        for rank in malicious_ranks:
            create_fedml_config(args, rank, True)

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


def generate_client_ranks(benign_clients, malicious_clients):
    client_ranks = list(range(1, benign_clients + malicious_clients + 1))
    # Shuffle clients randomly (a seed can be set)
    random.shuffle(client_ranks)
    benign_ranks = client_ranks[:benign_clients]
    malicious_ranks = client_ranks[benign_clients:]
    logger.info(f'Benign clients - {benign_ranks}')
    logger.info(f'Malicious clients - {malicious_ranks}')
    os.makedirs(CONFIG_OUT, exist_ok=True)
    with open(os.path.join(BENIGN_CONFIG_OUT), "w") as f:
        for rank in benign_ranks:
            f.write(f'{rank},')
    with open(os.path.join(MAL_CONFIG_OUT), "w") as f:
        for rank in malicious_ranks:
            f.write(f'{rank},')
    return benign_ranks, malicious_ranks


def generate_router_files(args, benign_ranks, malicious_ranks, dev=False):
    generate_router_image(args, dev)


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
        shutil.copy2(FADO_DEFAULT_MODEL_PATH, os.path.join(client_user_path, 'get_model.py'))


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


def create_ipconfig(ipconfig_out, rank, num_clients):
    """ Creates ipconfig file that tells FedML the IP of each node

        Parameters:
            ipconfig_out(str): Path for writing the ipconfig file
            rank: Rank of client
    """
    path = Path(os.path.join(ipconfig_out, f'user_{rank}', 'grpc_ipconfig.csv'))
    os.makedirs(path.parent.absolute(), exist_ok=True)
    with open(path, 'w') as f:
        base_server_ip = ipaddress.IPv4Address('10.2.1.0')
        base_client_ip = ipaddress.IPv4Address('10.1.1.0')
        f.write('receiver_id,ip\n')
        f.write(f'0,{base_server_ip}\n')
        for rank in range(1, num_clients + 1):
            f.write(f'{rank},{base_client_ip}\n')
            base_client_ip += 1


def generate_compose(dataset, docker_compose_out, n_clients, using_gpu=False):
    """
    Loads a default compose file and generates 'number_clients' of client services

        Parameters:
            docker_compose_out (str): Path of the output for the docker compose

        Returns:
            Two lists of integers representing the benign_ranks and malicious_ranks
    """
    import copy

    # Load the default docker compose
    docker_compose = load_base_compose(using_gpu)

    # Generate benign compose services
    base = docker_compose['services']['clients']
    client_compose = copy.deepcopy(base)

    # Add all volumes manually to give possibility to customize FADO home folder...
    client_compose['volumes'] = list()
    client_compose['volumes'] += [f'{PARTITION_DATA_FOLDER}/{dataset}/clients:/app/data']
    client_compose['volumes'] += [f'{CONFIG_OUT}:/app/config']
    client_compose['volumes'] += [f'{CERTS_OUT}:/app/certs']
    client_compose['volumes'] += [f'{ATTACK_DIRECTORY}:/app/attack:rw']
    client_compose['volumes'] += [f'{DEFENSE_DIRECTORY}:/app/defense:rw']
    client_compose['volumes'] += [f'{LOGS_DIRECTORY}:/app/logs:rw']


    client_compose['environment'] += [f'N_INTERFACES={n_clients}']
    docker_compose['services'][f'clients'] = client_compose

    # Customize volume for data in server
    # Add all volumes to give possibility to customize FADO home folder
    docker_compose['services']['server']['volumes'] = list()
    docker_compose['services']['server']['volumes'] += [f'{PARTITION_DATA_FOLDER}/{dataset}/server:/app/data/user_0']
    docker_compose['services']['server']['volumes'] += [f'{CONFIG_OUT}:/app/config']
    docker_compose['services']['server']['volumes'] += [f'{CERTS_OUT}:/app/certs']
    docker_compose['services']['server']['volumes'] += [f'{ATTACK_DIRECTORY}:/app/attack:rw']
    docker_compose['services']['server']['volumes'] += [f'{DEFENSE_DIRECTORY}:/app/defense:rw']
    docker_compose['services']['server']['volumes'] += [f'{LOGS_DIRECTORY}:/app/logs:rw']
    docker_compose['services']['server']['environment'] = []
    docker_compose['services']['server']['environment'] += [f'N_INTERFACES={n_clients}']

    # Same as above but for the router...
    docker_compose['services']['router']['volumes'] = list()
    docker_compose['services']['router']['volumes'] += [f'{CONFIG_OUT}:/app/config']


    with open(docker_compose_out, 'w') as f:
        yaml.dump(docker_compose, f, sort_keys=False)


def create_fedml_config(args, rank, malicious=False):
    """ Generates fedml_config files for FedML nodes

    Parameters:
        args: Arguments of FADO
        rank: client rank
        malicious: if this fedml_config is malicious

    """
    file_path = os.path.abspath(os.path.realpath(FEDML_CONFIG_FILE_PATH))

    # Load base docker compose file
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    client_num = args.benign_clients + args.malicious_clients
    fedml_config_out = os.path.join(CONFIG_OUT, f'user_{rank}')
    os.makedirs(fedml_config_out, exist_ok=True)

    if malicious:
        # maybe throw an exception, what if 'client_attack_spec' is not defined?
        # TODO: user has to be alerted
        if 'client_attack_spec' in args:
            config['attack_args'] = {}
            config['attack_args']['client_attack_spec'] = args.client_attack_spec
    else:
        if 'defense_spec' in args:
            config['defense_args'] = {}
            config['defense_args']['defense_spec'] = args.defense_spec
        if 'target_class' in args:
            config['monitor'] = {}
            config['monitor']['target_class'] = args.target_class

    config['data_args']['data_cache_dir'] = f'./data/user_{rank}'
    config['comm_args']['grpc_ipconfig_path'] = f'./config/user_{rank}/grpc_ipconfig.csv'
    config['common_args']['random_seed'] = args.random_seed
    config['train_args']['client_num_in_total'] = client_num
    config['train_args']['client_num_per_round'] = args.clients_per_round
    config['train_args']['comm_round'] = args.rounds
    config['train_args']['epochs'] = args.epochs
    config['train_args']['batch_size'] = args.batch_size
    config['train_args']['client_optimizer'] = args.client_optimizer
    config['device_args']['worker_num'] = client_num
    config['data_args']['dataset'] = args.dataset
    config['model_args']['model'] = args.model
    if "encrypt_comm" in args:
        config['comm_args']['encrypt'] = args.encrypt_comm
    with open(os.path.join(fedml_config_out, 'fedml_config.yaml'), 'w') as f:
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

    # Put inside the docker compose file the client and server base files
    for service in ['server', 'clients', 'router']:
        with open(dir_path + os.path.sep + docker_compose['services'][service]['compose-file'], 'r') as file:
            compose = yaml.load(file, Loader=yaml.FullLoader)
            docker_compose['services'][service].pop('compose-file')
            docker_compose['services'][service] = compose      

    if using_gpu:
        with open(dir_path + os.path.sep + 'gpu_compose.yaml', 'r') as file:
            gpu_compose = yaml.load(file, Loader=yaml.FullLoader)
            docker_compose['services']['clients']['deploy'] = gpu_compose['deploy']
            docker_compose['services']['server']['deploy'] = {}
            docker_compose['services']['server']['deploy']['resources'] = gpu_compose['deploy']['resources']
    return docker_compose
