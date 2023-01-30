import json
import os
import shutil
import subprocess

from fado import leaf
from fado.constants import ALL_DATA_FOLDER, FADO_DIR, TEMP_DIRECTORY

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split

from concurrent.futures import ThreadPoolExecutor, as_completed


def leaf_downloader(args):

    """
    The LEAF's module inside FADO contains, for each different dataset,
    the scripts that download and process the data.

    This method copies those scripts into a temporary directory that is created
    in the current working directory called 'temp_leaf'. It is not a good practice
    to download and process data directly in the module's directory.

    After executing the script that downloads and generates the data (preprocess),
    the data that is generated is copied to the data directory used by FADO (./data/all).

    Data from different datasets is organized inside ./data/all under different directories
    to provide agility in experimenting with different datasets without having to download
    and process them over and over again.

    """

    dataset = args.dataset
    dataset_rate = args.dataset_rate
    data_distribution = args.data_distribution

    data_fado = os.path.join(
        ALL_DATA_FOLDER, dataset, data_distribution, f"frac_{str(dataset_rate)[2:]}"
    )

    if not os.path.exists(data_fado):
        os.makedirs(data_fado, exist_ok=True)

    # Getting LEAF's module path
    leaf_path = os.path.dirname(leaf.__file__)

    # Copying LEAF's utils folder to temp directory
    shutil.copytree(
        os.path.join(leaf_path, "utils"),
        os.path.join(TEMP_DIRECTORY, "utils"),
        dirs_exist_ok=True,
    )
    # Copying LEAF's dataset folder to temp directory
    shutil.copytree(
        os.path.join(leaf_path, dataset),
        os.path.join(TEMP_DIRECTORY, dataset),
        dirs_exist_ok=True,
    )

    curr_dir = os.getcwd()  # will be useful to return to current directory
    os.chdir(os.path.join(TEMP_DIRECTORY, dataset))  # changing current directory

    # Calling LEAF script
    print(f"./preprocess.sh -s {data_distribution} --sf {dataset_rate} -k 0 -t sample")
    subprocess.call(
        [
            "./preprocess.sh",
            f"-s {data_distribution}",
            f"--sf {dataset_rate}",
            f"-k 0",
            f"-t sample",
        ]
    )

    # Remove unnecessary folders
    data_path = os.path.join(TEMP_DIRECTORY, dataset, "data")
    subfolders = [
        f.path
        for f in os.scandir(data_path)
        if f.is_dir() and f.name not in ["train", "test", "all_data"]
    ]
    for subfolder in subfolders:
        shutil.rmtree(subfolder, ignore_errors=True)

    # Copying train and test folder to the .{all_data_folder}/{dataset} folder
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    shutil.copytree(train_dir, os.path.join(data_fado, "train"), dirs_exist_ok=True)
    shutil.copytree(test_dir, os.path.join(data_fado, "test"), dirs_exist_ok=True)

    shutil.rmtree(os.path.join(TEMP_DIRECTORY, dataset, "meta"))
    shutil.rmtree(os.path.join(TEMP_DIRECTORY, dataset, "data", "train"))
    shutil.rmtree(os.path.join(TEMP_DIRECTORY, dataset, "data", "test"))

    os.chdir(curr_dir)


def torchvision_downloader(args):

    curr_dir = os.getcwd()  # will be useful to return to current directory

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset = args.dataset

    if dataset in ['cifar10', 'cifar100']:
        dataset_temp_dir = os.path.join(TEMP_DIRECTORY, 'cifar')
    else:
        dataset_temp_dir = os.path.join(TEMP_DIRECTORY, 'mnist')


    if not os.path.exists(dataset_temp_dir):
        os.makedirs(dataset_temp_dir, exist_ok=True)

    os.chdir(dataset_temp_dir)

    if dataset == "cifar10":

        training_dataset = datasets.CIFAR10(
            root=dataset_temp_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=dataset_temp_dir, train=False, download=True, transform=transform_test
        )

        training_subsets = random_split(training_dataset, [1000] * 50)
        test_subsets = random_split(test_dataset, [200] * 50)

    elif dataset == "cifar100":
        training_dataset = datasets.CIFAR100(
            root=dataset_temp_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=dataset_temp_dir, train=False, download=True, transform=transform_test
        )

        training_subsets = random_split(training_dataset, [1000] * 50)
        test_subsets = random_split(test_dataset, [200] * 50)
        
    elif dataset == "mnist":
        training_dataset = datasets.MNIST(
            root=dataset_temp_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        test_dataset = datasets.MNIST(
            root=dataset_temp_dir,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        training_subsets = random_split(training_dataset, [600] * 100)
        test_subsets = random_split(test_dataset, [100] * 100)

    
    # Splitting process for training, with previous verification

    dataset_training_output = os.path.join(ALL_DATA_FOLDER, dataset, 'iid', 'train', '')

    if not os.path.exists(dataset_training_output):

        print(f"Dividing {dataset} into training subsets...")

        for i, subset in enumerate(training_subsets):
            user_tr_data = []
            for j in range(len(subset)):
                user_tr_data.append(subset[j][0].numpy().tolist())
                
            user_tr_label = [subset[j][1] for j in range(len(subset))]

            all_data = {
                "x": user_tr_data,
                "y": user_tr_label
            }

            os.makedirs(os.path.dirname(dataset_training_output), exist_ok=True)
            user_data = {}
            user_data[f'user_{i}'] = all_data
            with open(os.path.join(dataset_training_output, f'all_data_{i}_iid_train.json'), "w") as outfile:
                json.dump({'user_data': user_data}, outfile)
            print(f'all_data_{i}_iid_train.json')   

    else:
        print("Training dataset already generated, proceeding.")

    # Splitting process for test, with previous verification

    dataset_test_output = os.path.join(ALL_DATA_FOLDER, dataset, 'iid', 'test', '')

    if not os.path.exists(dataset_test_output):

        print("Dividing CIFAR10 into test subsets...")

        for i, subset in enumerate(test_subsets):
            user_te_data = []
            for j in range(len(subset)):
                user_te_data.append(subset[j][0].numpy().tolist())

            user_te_label = [subset[j][1] for j in range(len(subset))]

            all_data = {
                "x": user_te_data,
                "y": user_te_label
            }

            os.makedirs(os.path.dirname(dataset_test_output), exist_ok=True)
            user_data = {}
            user_data[f'user_{i}'] = all_data
            with open(os.path.join(dataset_test_output, f'all_data_{i}_iid_test.json'), "w") as outfile:
                json.dump({'user_data': user_data}, outfile)
            print(f'all_data_{i}_iid_test.json')
    else:
        print("Test dataset already generated, proceeding.")

    os.chdir(curr_dir)