import argparse
import os
import subprocess
import sys
import shutil

from fado.arguments.arguments import AttackArguments
from fado.constants import ALL_DATA_FOLDER, FADO_DIR, LOGS_DIRECTORY, PARTITION_DATA_FOLDER, FADO_DEFAULT_CONFIG_FILE_PATH
from fado.orchestrate import prepare_orchestrate
from fado.data.downloader import leaf_executor
from fado.data.data_splitter import split_data


def data(args):
    print("Executing LEAF...")
    leaf_executor(args)


def partitions(args):
    print("Splitting data...")
    split_data(
        args.dataset,
        ALL_DATA_FOLDER,
        PARTITION_DATA_FOLDER,
        args.benign_clients + args.malicious_clients
    )


def compose(args, config, dev=True):
    prepare_orchestrate(config, args, dev)


def run():
    print("Deploying...")
    os.chdir(FADO_DIR)
    subprocess.run(['docker', 'compose', 'down'])
    subprocess.run(['docker', 'compose', 'build'])
    subprocess.run(['docker', 'compose', 'up'])  # up/stack ?


def clean():
    print("Cleaning...")
    shutil.rmtree(PARTITION_DATA_FOLDER)
    shutil.rmtree(LOGS_DIRECTORY)
    
def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='yaml_file', type=str, help='Specify a custom yaml configuration file',
                        required=False)
    mode_parser = parser.add_subparsers(dest="mode", required=False)

    build_parser = mode_parser.add_parser('build')
    mode_parser.add_parser('run')
    mode_parser.add_parser('clean')

    parser.add_argument('-d', dest='dataset', type=str, choices=['femnist', 'shakespeare', 'sent140'], required=False)
    parser.add_argument('-dr', dest='dataset_rate', help='Fraction of the dataset', default='0.05', type=float, required=False)
    parser.add_argument('-nb', dest='number_benign', type=int, required=False)
    parser.add_argument('-nm', dest='number_malicious', type=int, required=False)

    build_mode_parser = build_parser.add_subparsers(dest="build_mode")
    build_mode_parser.add_parser('data')
    build_mode_parser.add_parser('partitions')
    build_mode_parser.add_parser('compose')

    return parser.parse_args(args)


def cli():
    os.umask(0)

    args = parse_args(sys.argv[1:])

    config_file = args.yaml_file if args.yaml_file else FADO_DEFAULT_CONFIG_FILE_PATH

    fado_arguments = AttackArguments(config_file)

    if args.number_benign:
        fado_arguments.set_argument('number_benign', args.number_benign)
    if args.number_malicious:
        fado_arguments.set_argument('malicious_clients', args.number_malicious)
    if args.dataset:
        fado_arguments.set_argument('dataset', args.dataset)
    if args.dataset_rate:
        fado_arguments.set_argument('dataset_rate', args.dataset_rate)

    if args.mode == 'build':
        build_mode = args.build_mode

        if build_mode == 'data':
            data(fado_arguments)
        elif build_mode == 'partitions':
            partitions(fado_arguments)
        elif build_mode == 'compose':
            # generate_compose on orchestrator.py
            compose(fado_arguments, config_file, True)
        else:
            data(fado_arguments)
            partitions(fado_arguments)
            compose(fado_arguments, config_file, True)

    elif args.mode == 'run':
        run()
    elif args.mode == 'clean':
        clean()
    else:
        data(fado_arguments)
        partitions(fado_arguments)
        compose(fado_arguments, config_file, True)
        run()


if __name__ == '__main__':
    cli()