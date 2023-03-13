import argparse
import itertools
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from threading import Thread

from fado.constants import *
from fado.cli.arguments.arguments import FADOArguments
from fado.runner import server_run, clients_run
from fado.runner.output.table import generate_table

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'builder'})


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='yaml_file', type=str, help='Specify a custom yaml configuration file',
                        required=False)
    mode_parser = parser.add_subparsers(dest="mode", required=False)

    build_parser = mode_parser.add_parser('build')
    mode_parser.add_parser('run')
    mode_parser.add_parser('table')

    parser.add_argument('-d', dest='dataset', type=str, choices=DATASETS, required=False)
    parser.add_argument('-dr', dest='dataset_rate', help='Fraction of the dataset', type=float,
                        required=False)
    parser.add_argument('-dd', dest='data_distribution', help='Data distribution', type=str, required=False)
    parser.add_argument('-nb', dest='number_benign', type=int, required=False)
    parser.add_argument('-nm', dest='number_malicious', type=int, required=False)
    parser.add_argument('--dev', dest='development', action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument('--no-docker', dest='docker', action='store_false', default=True)

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


def create_networks():
    subprocess.run(['docker', 'network', 'create', 'clients-network'])
    subprocess.run(['docker', 'network', 'create', 'server-network'])


def run_clients(fado_args, dev_mode, docker, add_flags):
    if not docker:
        clients_run.main()
        return

    # Start clients container
    subprocess.run(['docker', 'run', '-d', '-w', '/app', '--name', 'fado-clients', '--cap-add=NET_ADMIN',
                    '--network', 'clients-network'] + add_flags +
                   ['ralexandre00/fado-node:latest', 'bash', '-c', 'tail -f /dev/null'])

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
                    'python3 -m fado.runner.communication.config.config_clients_network'])
    subprocess.run(['docker', 'exec', 'fado-clients', '/bin/bash', '-c', 'python3 -m fado.runner.clients_run'])


def run_server(fado_args, dev_mode, docker, add_flags):
    if not docker:
        server_run.main()
        return

    # Start server container
    subprocess.run(['docker', 'run', '-d', '-w', '/app', '--name', 'fado-server', '--cap-add=NET_ADMIN',
                    '-v', f'{LOGS_DIRECTORY}:/app/logs', '-v', f'{RESULTS_DIRECTORY}:/app/results',
                    '--network', 'server-network'] + add_flags +
                   ['ralexandre00/fado-node:latest', 'bash', '-c', 'tail -f /dev/null'])

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
                    'python3 -m fado.runner.communication.config.config_server_network'])
    subprocess.run(['docker', 'exec', 'fado-server', '/bin/bash', '-c', 'python3 -m fado.runner.server_run'])
    return


def run_router(fado_args, dev_mode, docker, add_flags):
    if not docker:
        # Do nothing
        return

    # Start server container
    subprocess.run(['docker', 'run', '-d', '-w', '/app', '--name', 'fado-router', '--cap-add=NET_ADMIN',
                    '--network', 'server-network'] + add_flags +
                   ['ralexandre00/fado-router:latest', 'bash', '-c', 'tail -f /dev/null'])
    subprocess.run(['docker', 'network', 'connect', 'clients-network', 'fado-router'])

    # Send fado_config and data to container
    subprocess.run(['docker', 'cp', f'{FADO_CONFIG_OUT}', 'fado-router:/app/config/fado_config.yaml'])
    data_path = os.path.join(ALL_DATA_FOLDER, fado_args.dataset)
    subprocess.run(['docker', 'cp', os.path.join(data_path, 'target_test_attacker'), 'fado-router:/app/data'])

    if dev_mode:
        # Install current fado
        subprocess.run(['docker', 'cp', f'{FADO_SRC}', 'fado-router:/app/fado'])
        subprocess.run(['docker', 'exec', 'fado-router', '/bin/bash', './run_dev.sh'], stdout=subprocess.DEVNULL)

    # Start router
    subprocess.run(['docker', 'exec', 'fado-router', '/bin/bash', '-c',
                    'python3 -m fado.runner.communication.config.config_router_network'])
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


def run(fado_args, dev_mode=False, docker=True):
    container_flags = []
    if fado_args.use_gpu:
        container_flags = ['--gpus', 'all']
    if not docker:
        os.environ['FADO_DATA_PATH'] = os.path.join(ALL_DATA_FOLDER, fado_args.dataset)
        os.environ['FADO_CONFIG_PATH'] = FADO_CONFIG_OUT
        os.environ['LOG_FILE_PATH'] = LOGS_DIRECTORY
        os.environ['RESULTS_FILE_PATH'] = RESULTS_DIRECTORY
        os.environ['SERVER_IP'] = 'localhost'
    try:
        if docker:
            create_networks()
        Thread(target=run_router, args=(fado_args, dev_mode, docker, container_flags,), daemon=True).start()
        Thread(target=run_server, args=(fado_args, dev_mode, docker, container_flags), daemon=True).start()
        t = Thread(target=run_clients, args=(fado_args, dev_mode, docker, container_flags,), daemon=True)
        t.start()
        t.join()
    finally:
        if docker:
            stop_server()
            stop_router()
            stop_clients()


def move_files_to_fado_home(config_file):
    os.makedirs(CONFIG_OUT, exist_ok=True)
    os.makedirs(LOGS_DIRECTORY, exist_ok=True)
    shutil.copyfile(config_file, FADO_CONFIG_OUT)


def run_multiple(fado_arguments, development, docker):
    # Create a dictionary with all possible combinations of configs to vary
    vary_list = list(itertools.product(*fado_arguments.vary.values()))
    temp_output = os.path.join(TEMP_DIRECTORY, 'fado_config.yaml')
    os.makedirs(TEMP_DIRECTORY, exist_ok=True)

    # For each combination execute fado run
    for experiment in vary_list:
        for i, vary_key in enumerate(fado_arguments.vary.keys()):
            fado_arguments.set_argument(vary_key, experiment[i])

        # Save current experiment fado_config file in temp folder
        fado_arguments.save_to_file(temp_output)
        logger.info(f"Running experiment {experiment}")

        fado_arguments_experiment = FADOArguments(temp_output)

        move_files_to_fado_home(temp_output)
        download_data(fado_arguments_experiment)
        shape_data(fado_arguments_experiment)
        run(fado_arguments_experiment, development, docker)
    os.remove(temp_output)


def cli():
    args = parse_args(sys.argv[1:])

    if args.yaml_file:
        config_file = args.yaml_file
    else:
        fado_config_env = os.getenv('FADO_CONFIG')
        config_file = fado_config_env if fado_config_env else FADO_DEFAULT_CONFIG_FILE_PATH

    fado_arguments = FADOArguments(config_file)

    # docker pull ralexandre00/fado-node-requirements

    if args.mode == 'build':
        build_mode = args.build_mode
        move_files_to_fado_home(config_file)

        if build_mode == 'download':
            download_data(fado_arguments)
        elif build_mode == 'shape':
            shape_data(fado_arguments)
        else:
            download_data(fado_arguments)
            shape_data(fado_arguments)

    elif args.mode == 'run':
        if 'vary' in fado_arguments:
            run_multiple(fado_arguments, args.development, args.docker)
            return

        move_files_to_fado_home(config_file)
        run(fado_arguments, args.development, args.docker)
    elif args.mode == 'table':
        generate_table(100)
        # clean()
    else:
        download_data(fado_arguments)
        shape_data(fado_arguments)
        run(fado_arguments, args.development, args.docker)
