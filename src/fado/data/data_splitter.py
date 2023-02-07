import json
import logging
import os
import pickle
import numpy as np

from fado.constants import CONFIG_HASH, ALL_DATA_FOLDER, PARTITION_DATA_FOLDER
from fado.crypto.hash_verifier import file_changed, write_file_hash
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split


__all__ = ['split_data']

logger = logging.getLogger('fado')


def gather_test_target(test_target_user, data):
    test_target_user.setdefault('x', [])
    test_target_user.setdefault('y', [])
    test_target_user['x'].append(data[0])
    test_target_user['y'].append(data[1])


def split_data(args, target_class=None):
    """Takes a folder with data json files and created 'num_users' folder each with specific client data

        Assumes:
            the data in the .json files have pairs ('user_id','user_data')

        Parameters:
            all_data_folder: folder with two folder, train and test data each with data json files
            partition_data_folder: output folder for client partition json files
            num_users: number of users to generate partitions for
            target_class: class for creating a specific partition for testing

    """

    dataset = args.dataset
    dataset_rate = args.dataset_rate
    num_users = args.benign_clients + args.malicious_clients
    data_distribution = args.data_distribution


    if dataset in ['cifar10', 'cifar100', 'mnist']:

        if num_users > 50:
            raise Exception("CIFAR datasets do not support more than 50 users.")

        data_distribution = 'iid'

    for t in ["train", 'test']:
        all_data = {}
        server_data = {}
        data_files = os.listdir(os.path.join(ALL_DATA_FOLDER, dataset, data_distribution, f"frac_{str(dataset_rate)[2:]}", t))
        # Get all json files from all_data_folder
        data_files = [f for f in data_files if f.endswith(".json")]
        for f in data_files:
            with open(os.path.join(ALL_DATA_FOLDER, dataset, data_distribution, f"frac_{str(dataset_rate)[2:]}", t, f), 'r') as file:
                j = json.load(file)
            all_data.update(j['user_data'])

        users = list(all_data.keys())
        users.sort()
        users = users[:num_users]

        for i, user_id in enumerate(users, start=1):
            os.makedirs(os.path.dirname(os.path.join(PARTITION_DATA_FOLDER, dataset, 'clients', f'user_{i}', t, '')),
                        exist_ok=True)
            # Create a dictionary with user ids and empty lists
            user_data = {k: {} for k in users}
            # Set the corresponding user list
            user_data[user_id] = all_data[user_id]
            server_data[user_id] = all_data[user_id]
            with open(os.path.join(PARTITION_DATA_FOLDER, dataset, 'clients', f'user_{i}', t, 'data.json'), "w") as outfile:
                json.dump({'user_data': user_data}, outfile)

        os.makedirs(os.path.dirname(os.path.join(PARTITION_DATA_FOLDER, dataset, 'server', t, '')), exist_ok=True)
        # The server uses the train data of the clients training and all the test available
        if t == 'train':
            with open(os.path.join(PARTITION_DATA_FOLDER, dataset, 'server', t, 'data.json'), "w") as outfile:
                json.dump({'user_data': server_data}, outfile)
        else:
            with open(os.path.join(PARTITION_DATA_FOLDER, dataset, 'server', t, 'data.json'), "w") as outfile:
                json.dump({'user_data': all_data}, outfile)

    # Create a test dataset to test for backdoor attacks
    found = 0
    if target_class is not None:
        test_target = {}
        for user, data in all_data.items():
            # d[1] is y
            result = list(filter(lambda d: d[1] == target_class, zip(data['x'], data['y'])))
            test_target[user] = {}
            any(gather_test_target(test_target[user], d) for d in result)
            found += len(result)
        os.makedirs(os.path.join(PARTITION_DATA_FOLDER, dataset, 'server', 'target_test'), exist_ok=True)
        if found == 0:
            logger.error("Target class has no examples")
            exit(-1)
        with open(os.path.join(PARTITION_DATA_FOLDER, dataset, 'server', 'target_test', 'data.json'), "w") as outfile:
            json.dump({'user_data': test_target}, outfile)
