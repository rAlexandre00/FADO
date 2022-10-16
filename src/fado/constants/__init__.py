import os

from fado import orchestrate, docker

orchestrate_module_dir = os.path.dirname(orchestrate.__file__)
docker_module_dir = os.path.dirname(docker.__file__)

GENERAL_COMPOSE_FILE_PATH = os.path.join(orchestrate_module_dir, 'base_files', 'general_compose.yaml')
FEDML_CONFIG_FILE_PATH = os.path.join(orchestrate_module_dir, 'base_files', 'fedml_config.yaml')
BEN_CLIENT_PATH = os.path.join(docker_module_dir, 'benign-client')
MAL_CLIENT_PATH = os.path.join(docker_module_dir, 'malicious-client')
SERVER_PATH = os.path.join(docker_module_dir, 'server')
