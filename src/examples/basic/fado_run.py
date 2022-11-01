import argparse
import os
import subprocess
import sys
import shutil

from fado.arguments.arguments import AttackArguments
from fado.constants import ALL_DATA_FOLDER, FADO_DIR, PARTITION_DATA_FOLDER
from fado.orchestrate import prepare_orchestrate
from fado.data.downloader import leaf_executor
from fado.data.data_splitter import split_data

def data(args):
    leaf_executor(args)

def partitions(args):
    split_data(
        args.dataset, 
        ALL_DATA_FOLDER, 
        PARTITION_DATA_FOLDER, 
        args.benign_clients + args.malicious_clients
    )

def compose():
    pass

def run():
    os.chdir(FADO_DIR)
    subprocess.run(['docker', 'compose', 'build'])
    subprocess.run(['docker', 'compose', 'up'])  # stack?

def clean():
    shutil.rmtree(FADO_DIR)
    
def parse_args(args):
    parser = argparse.ArgumentParser()
    mode_parser = parser.add_subparsers(dest="mode", required=True)

    build_parser = mode_parser.add_parser('build')
    mode_parser.add_parser('run')
    mode_parser.add_parser('clean')

    build_parser.add_argument('-d', dest='dataset', type=str, choices=['femnist', 'shakespeare', 'sent140'], required=False)
    build_parser.add_argument('-nb', dest='number_benign', type=int, required=False)
    build_parser.add_argument('-nm', dest='number_malicious', type=int, required=False)

    build_mode_parser = build_parser.add_subparsers(dest="build_mode")
    build_mode_parser.add_parser('data')
    build_mode_parser.add_parser('partitions')
    build_mode_parser.add_parser('compose')

    return parser.parse_args(args)

if __name__ == '__main__':

    args = parse_args(sys.argv[1:])

    config_path = 'config/fado_config.yaml'
    fado_arguments = AttackArguments(config_path) 

    if args.number_benign:
        fado_arguments['benign_clients'] = args.number_benign
    if args.number_malicious:
        fado_arguments['malicious_clients'] = args.number_malicious
    if args.dataset:
        fado_arguments['dataset'] = args.dataset

    if args.mode == 'build':
        build_mode = args.build_mode

        if build_mode == 'data':
            data(fado_arguments)
        elif build_mode == 'partitions':
            partitions(fado_arguments)
        elif build_mode == 'compose':
            # generate_compose on orchestrator.py
            compose()
        else:
            data(fado_arguments)
            partitions(fado_arguments)
            compose()

    elif args.mode == 'run':
        run()
    elif args.mode == 'clean':
        clean()