import argparse
import os
import signal
import subprocess
import sys
import shutil

from fado.arguments.arguments import AttackArguments
from fado.constants import *
from fado.orchestrate import prepare_orchestrate
from fado.data.downloader import leaf_downloader, torchvision_downloader
from fado.data.data_splitter import split_data


def data(args):

    dataset = args.dataset

    if dataset not in DATASETS:
        raise Exception(f"Dataset {dataset} not supported! Choose one of the following: {DATASETS}")

    if dataset in LEAF_DATASETS:
        print("Executing LEAF...")
        leaf_downloader(args)
    else:
        print("Executing Torch vision Downloader...")
        torchvision_downloader(args)

    partitions(args)


def partitions(args):
    print("Splitting data...")
    target_class = None
    if 'target_class' in args:
        target_class = args.target_class
    split_data(
        args,
        target_class=target_class
    )


def compose(args, config, dev=True):
    prepare_orchestrate(config, args, dev)


def run():
    print("Deploying...")
    os.chdir(FADO_DIR)
    subprocess.run(['docker', 'compose', 'down'])
    subprocess.run(['docker', 'compose', 'build'])
    try:
        p = subprocess.Popen(['docker', 'compose', 'up', '--remove-orphans'])
        p.wait()
    except KeyboardInterrupt:
        try:
            p.send_signal(signal.SIGINT)
            p.wait()
        except:
            pass


def clean():
    print("Cleaning...")
    shutil.rmtree(PARTITION_DATA_FOLDER, ignore_errors=True)
    shutil.rmtree(ATTACK_DIRECTORY, ignore_errors=True)
    shutil.rmtree(DEFENSE_DIRECTORY, ignore_errors=True)
    shutil.rmtree(CONFIG_OUT, ignore_errors=True)
    shutil.rmtree(CERTS_OUT, ignore_errors=True)
    shutil.rmtree(IMAGES_PATH, ignore_errors=True)
    shutil.rmtree(TENSORBOARD_DIRECTORY, ignore_errors=True)
    shutil.rmtree(LOGS_DIRECTORY, ignore_errors=True)
    os.remove(CONFIG_HASH)
    os.remove(DOCKER_COMPOSE_OUT)


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', dest='yaml_file', type=str, help='Specify a custom yaml configuration file',
                        required=False)
    mode_parser = parser.add_subparsers(dest="mode", required=False)

    build_parser = mode_parser.add_parser('build')
    mode_parser.add_parser('run')
    mode_parser.add_parser('clean')

    parser.add_argument('-d', dest='dataset', type=str, choices=DATASETS, required=False)
    parser.add_argument('-dr', dest='dataset_rate', help='Fraction of the dataset', default='0.05', type=float,
                        required=False)
    parser.add_argument('-dd', dest='data_distribution', help='Data distribution', default='niid', type=str,
    required=False)
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
    if args.data_distribution:
        fado_arguments.set_argument('data_distribution', args.data_distribution)

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
            compose(fado_arguments, config_file, True)

    elif args.mode == 'run':
        run()
    elif args.mode == 'clean':
        clean()
    else:
        data(fado_arguments)
        compose(fado_arguments, config_file, True)
        run()


if __name__ == '__main__':
    cli()
