import os

import fado
from fado.builder import crypto, default

fado_module_dir = os.path.dirname(fado.__file__)
crypto_module_dir = os.path.dirname(crypto.__file__)
default_module_dir = os.path.dirname(default.__file__)

FADO_VERSION = "0.0.4"

FADO_DIR = os.getenv('FADO_HOME_FOLDER', os.path.join(os.path.expanduser("~"), '.fado'))
FADO_SRC = os.path.join(fado_module_dir)

CERTS_PATH = os.path.join(crypto_module_dir, 'certs')
FADO_DEFAULT_CONFIG_FILE_PATH = os.path.join(default_module_dir, 'fado_config.yaml')

CONFIG_HASH = os.path.join(FADO_DIR, '.config_hash')
LOGS_DIRECTORY = os.path.join(FADO_DIR, 'logs')
RESULTS_DIRECTORY = os.path.join(FADO_DIR, 'results')
TENSORBOARD_DIRECTORY = os.path.join(FADO_DIR, 'runs')
ATTACK_DIRECTORY = os.path.join(FADO_DIR, 'attack')
DEFENSE_DIRECTORY = os.path.join(FADO_DIR, 'defense')
CONFIG_OUT = os.path.join(FADO_DIR, 'config')
IMPORT_OUT = os.path.join(FADO_DIR, 'import')
FADO_CONFIG_OUT = os.path.join(CONFIG_OUT, 'fado_config.yaml')
BENIGN_CONFIG_OUT = os.path.join(CONFIG_OUT, 'benign_ranks.csv')
MAL_CONFIG_OUT = os.path.join(CONFIG_OUT, 'malicious_ranks.csv')
CERTS_OUT = os.path.join(FADO_DIR, 'certs')
DOCKER_COMPOSE_OUT = os.path.join(FADO_DIR, 'docker-compose.yml')
TEMP_DIRECTORY = os.path.join(FADO_DIR, 'temp')

ALL_DATA_FOLDER = os.path.join(FADO_DIR, 'data')

IMAGES_PATH = os.path.join(FADO_DIR, 'docker')
FEDML_IMAGE = os.path.join(IMAGES_PATH, 'client')
ROUTER_IMAGE = os.path.join(IMAGES_PATH, 'router')

LEAF_DATASETS = ['femnist', 'emnist', 'shakespeare', 'sent140']
NLAFL_DATASETS = ['nlafl_emnist', 'nlafl_fashionmnist', 'nlafl_dbpedia']
DATASETS = ['custom', 'cifar10', 'cifar100', 'mnist'] + LEAF_DATASETS + NLAFL_DATASETS

SERVER_IP = '10.0.0.2'
SERVER_PORT = 51000
SERVER_PUB_PORT = 51001
