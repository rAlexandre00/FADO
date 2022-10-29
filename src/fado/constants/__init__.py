import os

from fado import orchestrate, docker

orchestrate_module_dir = os.path.dirname(orchestrate.__file__)
docker_module_dir = os.path.dirname(docker.__file__)

FADO_DIR = os.path.join(os.path.expanduser("~"), '.fado')

GENERAL_COMPOSE_FILE_PATH = os.path.join(orchestrate_module_dir, 'base_files', 'general_compose.yaml')
FEDML_CONFIG_FILE_PATH = os.path.join(orchestrate_module_dir, 'base_files', 'fedml_config.yaml')
CLIENT_PATH = os.path.join(docker_module_dir, 'client')
ROUTER_PATH = os.path.join(docker_module_dir, 'router')
MAL_CLIENT_PATH = os.path.join(docker_module_dir, 'malicious-client')

CONFIG_HASH = os.path.join(FADO_DIR, '.config_hash')
LOGS_DIRECTORY = os.path.join(FADO_DIR, 'logs')
TENSORBOARD_DIRECTORY = os.path.join(FADO_DIR, 'runs')
ATTACK_DIRECTORY = os.path.join(FADO_DIR, 'attack')
DEFENSE_DIRECTORY = os.path.join(FADO_DIR, 'defense')
FEDML_BEN_CONFIG_OUT = os.path.join(FADO_DIR, 'config', 'fedml_config.yaml')
FEDML_MAL_CONFIG_OUT = os.path.join(FADO_DIR, 'config', 'fedml_config_malicious.yaml')
GRPC_CONFIG_OUT = os.path.join(FADO_DIR, 'config', 'grpc_ipconfig.csv')
CERTS_OUT = os.path.join(FADO_DIR, 'certs')
DOCKER_COMPOSE_OUT = os.path.join(FADO_DIR, 'docker-compose.yml')

ALL_DATA_FOLDER = os.path.join(FADO_DIR, 'data', 'all')
PARTITION_DATA_FOLDER = os.path.join(FADO_DIR, 'data', 'partitions')

IMAGES_PATH = os.path.join(FADO_DIR, 'docker')
FEDML_IMAGE = os.path.join(IMAGES_PATH, 'client')
ROUTER_IMAGE = os.path.join(IMAGES_PATH, 'router')
