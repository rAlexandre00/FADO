import os
from distutils.dir_util import copy_tree

import yaml
import shutil

from fado.arguments import AttackArguments
from fado.data import split_data
from fado.crypto import generate_self_signed_certs, generate_certs
from fado.constants import FEDML_CONFIG_FILE_PATH


def prepare_orchestrate(config_path):
    args = AttackArguments(config_path)

    # 1. Generate image files
    from fado.constants import CLIENT_PATH, MAL_CLIENT_PATH
    copy_tree(CLIENT_PATH, "./docker/client/")
    #copy_tree(MAL_CLIENT_PATH, "./docker/malicious-client/")
    shutil.copyfile(args.model_file, "./docker/client/get_model.py")
    #shutil.copyfile(args.model_file, "./docker/malicious-client/get_model.py")
    #shutil.copyfile(args.model_file, "./docker/server/get_model.py")

    # 2. Generate docker-compose file
    compose, client_ranks = generate_compose(args.benign_clients, args.malicious_clients)
    with open(args.docker_compose_out, 'w') as f:
        yaml.dump(compose, f, sort_keys=False)

    # 3. Create grpc_ipconfig file and fedml_config.yaml
    create_ipconfig(args.grpc_ipconfig_out, client_ranks, args.benign_clients)
    create_fedml_config(args)
    create_fedml_config(args, True)

    # 4. Generate tls certificates (if defined in attacks args)
    os.makedirs('./certs/', exist_ok=True)
    generate_self_signed_certs(out_key_file="./certs/ca-key.pem", out_cert_file="./certs/ca-cert.pem")
    generate_certs(out_key_file="./certs/server-key.pem", out_cert_file="./certs/server-cert.pem",
                   ca_key_file="./certs/ca-key.pem", ca_cert_file="./certs/ca-cert.pem")


    # 5. Split data for each client for train and test
    split_data(args.all_data_folder, args.partition_data_folder, args.benign_clients + args.malicious_clients)


def create_ipconfig(ipconfig_out, client_ranks, number_ben_clients):
    with open(ipconfig_out, 'w') as f:
        f.write('receiver_id,ip\n')
        f.write('0,fedml-server\n')
        for rank in client_ranks[:number_ben_clients]:
            f.write(f'{rank},fedml-benign-client-{rank}\n')
        for rank in client_ranks[number_ben_clients:]:
            f.write(f'{rank},fedml-malicious-client-{rank}\n')


def generate_compose(number_ben_clients, number_mal_clients):
    """
    Loads a default compose file and generates 'number_clients' of client services

        Parameters:
            base_file_path (str): Path of the base file
            number_ben_clients (int): Number of benign clients
            number_mal_clients (int): Number of malicious clients

        Returns:
            docker_compose (dict): Specifies the docker compose yaml
    """
    import random
    import copy
    # Load the default docker compose
    docker_compose = load_base_compose()

    client_ranks = list(range(1, number_ben_clients + number_mal_clients + 1))
    # TODO: Read from config file seed value
    # random.shuffle(client_ranks)

    # Generate benign compose services
    client_base = docker_compose['services']['fedml-client-benign']
    for client_rank in client_ranks[:number_ben_clients]:
        docker_compose['services'][f'fedml-beg-client-{client_rank}'] = copy.deepcopy(client_base)
        docker_compose['services'][f'fedml-beg-client-{client_rank}']['container_name'] += f'-{client_rank}'
        docker_compose['services'][f'fedml-beg-client-{client_rank}']['environment'] += [f'FEDML_RANK={client_rank}']
        docker_compose['services'][f'fedml-beg-client-{client_rank}']['volumes'] += [f'./data/partitions/user_{client_rank}:/app/data/']
    docker_compose['services'].pop('fedml-client-benign')

    # Generate malicious compose services
    client_base = docker_compose['services']['fedml-client-malicious']
    for client_rank in client_ranks[number_ben_clients:]:
        docker_compose['services'][f'fedml-mal-client-{client_rank}'] = copy.deepcopy(client_base)
        docker_compose['services'][f'fedml-mal-client-{client_rank}']['container_name'] += f'-{client_rank}'
        docker_compose['services'][f'fedml-mal-client-{client_rank}']['environment'] += [f'FEDML_RANK={client_rank}']
        docker_compose['services'][f'fedml-mal-client-{client_rank}']['volumes'] += [f'./data/partitions/user_{client_rank}:/app/data/']
    docker_compose['services'].pop('fedml-client-malicious')

    return docker_compose, client_ranks


def create_fedml_config(args, malicious=False):

    file_path = os.path.abspath(os.path.realpath(FEDML_CONFIG_FILE_PATH))

    # Load base docker compose file
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    client_num = args.benign_clients + args.malicious_clients

    if malicious:
        fedml_config_out = args.fedml_config_out_malicious
        # maybe throw an exception, what if 'attack_spec' is not defined?
        # user has to be alerted
        config['attack_args'] = {}
        config['attack_args']['attack_spec'] = args.attack_spec
    else:
        fedml_config_out = args.fedml_config_out

    config['train_args']['client_num_in_total'] = client_num
    with open(fedml_config_out, 'w') as f:
        yaml.dump(config, f, sort_keys=False)


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
    for service in ['fedml-server', 'fedml-client-benign', 'fedml-client-malicious']:
        with open(dir_path + os.path.sep + docker_compose['services'][service]['compose-file'], 'r') as file:
            compose = yaml.load(file, Loader=yaml.FullLoader)
            docker_compose['services'][service].pop('compose-file')
            docker_compose['services'][service] = compose
    return docker_compose
