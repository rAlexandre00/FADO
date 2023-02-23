import argparse
import logging
import subprocess
import sys
import time
from threading import Thread
from typing import Optional

import docker
from fado.constants import *
from fado.cli.arguments.arguments import FADOArguments

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'builder'})


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='yaml_file', type=str, help='Specify a custom yaml configuration file',
                        required=False)
    mode_parser = parser.add_subparsers(dest="mode", required=False)

    build_parser = mode_parser.add_parser('build')
    mode_parser.add_parser('run')
    mode_parser.add_parser('clean')

    parser.add_argument('-d', dest='dataset', type=str, choices=DATASETS, required=False)
    parser.add_argument('-dr', dest='dataset_rate', help='Fraction of the dataset', type=float,
                        required=False)
    parser.add_argument('-dd', dest='data_distribution', help='Data distribution', type=str, required=False)
    parser.add_argument('-nb', dest='number_benign', type=int, required=False)
    parser.add_argument('-nm', dest='number_malicious', type=int, required=False)
    parser.add_argument('--dev', dest='development', action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument('--local', dest='local', action=argparse.BooleanOptionalAction, required=False)

    build_mode_parser = build_parser.add_subparsers(dest="build_mode")
    build_mode_parser.add_parser('download')
    build_mode_parser.add_parser('shape')

    return parser.parse_args(args)


def download_data(fado_arguments):
    from fado.builder.data.download.nlafl_downloader import NLAFLDownloader
    dataset = fado_arguments.dataset

    if dataset not in DATASETS:
        raise Exception(f"Dataset {dataset} not supported! Choose one of the following: {DATASETS}")

    if dataset in LEAF_DATASETS:
        logger.info("Executing LEAF...")
        # TODO: LEAFDownloader().download()
    elif dataset in NLAFL_DATASETS:
        NLAFLDownloader().download()
    else:
        logger.info("Executing Torch vision Downloader...")
        # TODO: TorchVisionDownloader().download()


def shape_data(fado_arguments):
    from fado.builder.data.shape.nlafl_shaper import NLAFLShaper
    dataset = fado_arguments.dataset

    if dataset not in DATASETS:
        raise Exception(f"Dataset {dataset} not supported! Choose one of the following: {DATASETS}")

    if dataset in LEAF_DATASETS:
        logger.info("Executing LEAF...")
        # TODO: LEAFShaper().shape()
    elif dataset in NLAFL_DATASETS:
        NLAFLShaper().shape()
    else:
        logger.info("Executing Torch vision Shaper ...")
        # TODO: TorchVisionShaper.shape()


def run_client(fado_args, dev_mode, local, add_flags):
    if local:
        # TODO: Set FADO_DATA_PATH and FADO_CONFIG_PATH and start training without docker
        pass

    # Start clients container
    subprocess.run(['docker', 'run', '-d', '-w', '/app', '--name', 'fado-clients', '--cap-add=NET_ADMIN'] + add_flags +
                    ['ralexandre00/fado-node', 'bash', '-c', 'tail -f /dev/null'])

    # Send fado_config and data to container
    subprocess.run(['docker', 'cp', f'{FADO_CONFIG_OUT}', 'fado-clients:/app/config/fado_config.yaml'])
    client_data_path = os.path.join(ALL_DATA_FOLDER, fado_args.dataset, 'train')
    subprocess.run(['docker', 'cp', client_data_path, 'fado-clients:/app/data'])

    if dev_mode:
        # Install current fado
        subprocess.run(['docker', 'cp', f'{FADO_SRC}', 'fado-clients:/app/fado'])
        subprocess.run(['docker', 'exec', 'fado-clients', '/bin/bash', './run_dev.sh'], stdout=subprocess.DEVNULL)

    # Start clients
    subprocess.run(['docker', 'exec', 'fado-clients', '/bin/bash', '-c',
                    'export FADO_CONFIG_PATH=/app/config/fado_config.yaml && '
                    'export FADO_DATA_PATH=/app/data'])
    subprocess.run(['docker', 'exec', 'fado-clients', '/bin/bash', '-c', 'python3 -m fado.runner.communication.config.config_clients_network'])
    subprocess.run(['docker', 'exec', 'fado-clients', '/bin/bash', '-c', 'python3 -m fado.runner.clients_run'])


def run_server(fado_args, dev_mode, local, add_flags):
    if local:
        # TODO: Set FADO_DATA_PATH and FADO_CONFIG_PATH and start training without docker
        pass

    # Start server container
    subprocess.run(['docker', 'run', '-d', '-w', '/app', '--name', 'fado-server', '--cap-add=NET_ADMIN'] + add_flags +
                    ['ralexandre00/fado-node', 'bash', '-c', 'tail -f /dev/null'])

    # Send fado_config and data to container
    subprocess.run(['docker', 'cp', f'{FADO_CONFIG_OUT}', 'fado-server:/app/config/fado_config.yaml'])
    data_path = os.path.join(ALL_DATA_FOLDER, fado_args.dataset)
    subprocess.run(['docker', 'cp', os.path.join(data_path, 'train'), 'fado-server:/app/data'])
    subprocess.run(['docker', 'cp', os.path.join(data_path, 'test'), 'fado-server:/app/data'])
    subprocess.run(['docker', 'cp', os.path.join(data_path, 'target_test'), 'fado-server:/app/data'])

    if dev_mode:
        # Install current fado
        subprocess.run(['docker', 'cp', f'{FADO_SRC}', 'fado-server:/app/fado'])
        subprocess.run(['docker', 'exec', 'fado-server', '/bin/bash', './run_dev.sh'], stdout=subprocess.DEVNULL)

    # Start clients
    subprocess.run(['docker', 'exec', 'fado-server', '/bin/bash', '-c',
                    'export FADO_CONFIG_PATH=/app/config/fado_config.yaml && '
                    'export FADO_DATA_PATH=/app/data'])
    subprocess.run(['docker', 'exec', 'fado-server', '/bin/bash', '-c', 'python3 -m fado.runner.communication.config.config_server_network'])
    subprocess.run(['docker', 'exec', 'fado-server', '/bin/bash', '-c', 'python3 -m fado.runner.server_run'])
    return


def run_router(dev_mode, local):
    if local:
        # Do nothing
        return

    # Start server container
    subprocess.run(['docker', 'run', '-d', '-w', '/app', '--name', 'fado-router', '--cap-add=NET_ADMIN',
                    'ralexandre00/fado-router', 'bash', '-c', 'tail -f /dev/null'])

    # Send fado_config to container
    subprocess.run(['docker', 'cp', f'{FADO_CONFIG_OUT}', 'fado-router:/app/config/fado_config.yaml'])

    if dev_mode:
        # Install current fado
        subprocess.run(['docker', 'cp', f'{FADO_SRC}', 'fado-router:/app/fado'])
        subprocess.run(['docker', 'exec', 'fado-router', '/bin/bash', './run_dev.sh'], stdout=subprocess.DEVNULL)

    # Start router
    subprocess.run(['docker', 'exec', 'fado-router', '/bin/bash', '-c', 'python3 -m fado.runner.communication.config.config_router_network'])
    subprocess.run(['docker', 'exec', 'fado-router', '/bin/bash', '-c', 'python3 -m fado.runner.router_run'])
    return


def stop_server():
    subprocess.run(['docker', 'kill', 'fado-server'], stdout=subprocess.DEVNULL)
    subprocess.run(['docker', 'rm', 'fado-server'], stdout=subprocess.DEVNULL)


def stop_clients():
    subprocess.run(['docker', 'kill', 'fado-clients'], stdout=subprocess.DEVNULL)
    subprocess.run(['docker', 'rm', 'fado-clients'], stdout=subprocess.DEVNULL)


def stop_router():
    subprocess.run(['docker', 'kill', 'fado-router'], stdout=subprocess.DEVNULL)
    subprocess.run(['docker', 'rm', 'fado-router'], stdout=subprocess.DEVNULL)


def run(fado_args, dev_mode=False, local=False):
    container_flags = []
    if fado_args.use_gpu:
        container_flags = ['--gpus', 'all']
    try:
        Thread(target=run_server, args=(fado_args, dev_mode, local, container_flags,), daemon=True).start()
        Thread(target=run_router, args=(dev_mode, local,), daemon=True).start()
        Thread(target=run_client, args=(fado_args, dev_mode, local, container_flags), daemon=True).start()
        while True:
            time.sleep(100)
    except KeyboardInterrupt:
        stop_server()
        stop_router()
        stop_clients()


def cli():
    args = parse_args(sys.argv[1:])
    print(args.mode)

    if args.yaml_file:
        config_file = args.yaml_file
    else:
        fado_config_env = os.getenv('FADO_CONFIG')
        config_file = fado_config_env if fado_config_env else FADO_DEFAULT_CONFIG_FILE_PATH

    fado_arguments = FADOArguments(config_file)

    if args.mode == 'build':
        build_mode = args.build_mode

        if build_mode == 'download':
            download_data(fado_arguments)
        elif build_mode == 'shape':
            shape_data(fado_arguments)
        else:
            download_data(fado_arguments)
            shape_data(fado_arguments)

    elif args.mode == 'run':
        run(fado_arguments, args.development, args.local)
    elif args.mode == 'clean':
        clean()
    else:
        data(fado_arguments)
        compose(fado_arguments, config_file, True)
        run()


def is_container_running(container_name: str) -> Optional[bool]:
    """Verify the status of a container by it's name

    :param container_name: the name of the container
    :return: boolean or None
    """
    RUNNING = "running"
    # Connect to Docker using the default socket or the configuration
    # in your environment
    docker_client = docker.from_env()
    # Or give configuration
    # docker_socket = "unix://var/run/docker.sock"
    # docker_client = docker.DockerClient(docker_socket)

    try:
        container = docker_client.containers.get(container_name)
    except docker.errors.NotFound as exc:
        pass
    else:
        container_state = container.attrs["State"]
        return container_state["Status"] == RUNNING

