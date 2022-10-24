from fado.orchestrate import prepare_orchestrate

from download_data import download_mnist

download_mnist('data')
prepare_orchestrate('config/attack_config.yaml', dev=True)