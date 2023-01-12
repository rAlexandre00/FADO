import json
import logging
import os

from fado.constants import CONFIG_HASH
from fado.crypto.hash_verifier import file_changed, write_file_hash

__all__ = ['split_data']

logger = logging.getLogger('fado')


def gather_test_target(test_target_user, data):
    test_target_user.setdefault('x', [])
    test_target_user.setdefault('y', [])
    test_target_user['x'].append(data[0])
    test_target_user['y'].append(data[1])


def split_data(dataset, all_data_folder, partition_data_folder, num_users, target_class=None):
    """Takes a folder with data json files and created 'num_users' folder each with specific client data

        Assumes:
            the data in the .json files have pairs ('user_id','user_data')

        Parameters:
            all_data_folder: folder with two folder, train and test data each with data json files
            partition_data_folder: output folder for client partition json files
            num_users: number of users to generate partitions for
            target_class: class for creating a specific partition for testing

    """

    # config_changed = file_changed(config_path, CONFIG_HASH)
    # if not config_changed:
    #     logger.warning('Attack config has not changed. Will not split the data...')
    #     return
    # else:
    #     write_file_hash(config_path, CONFIG_HASH)

    for t in ["train", 'test']:
        all_data = {}
        server_data = {}
        data_files = os.listdir(os.path.join(all_data_folder, dataset, t))
        # Get all json files from all_data_folder
        data_files = [f for f in data_files if f.endswith(".json")]
        for f in data_files:
            with open(os.path.join(all_data_folder, dataset, t, f), 'r') as file:
                j = json.load(file)
            all_data.update(j['user_data'])

        users = list(all_data.keys())
        users.sort()
        users = users[:num_users]

        for i, user_id in enumerate(users, start=1):
            os.makedirs(os.path.dirname(os.path.join(partition_data_folder, dataset, 'clients', f'user_{i}', t, '')),
                        exist_ok=True)
            # Create a dictionary with user ids and empty lists
            user_data = {k: {} for k in users}
            # Set the corresponding user list
            user_data[user_id] = all_data[user_id]
            server_data[user_id] = all_data[user_id]
            with open(os.path.join(partition_data_folder, dataset, 'clients', f'user_{i}', t, 'data.json'), "w") as outfile:
                json.dump({'user_data': user_data}, outfile)

        os.makedirs(os.path.dirname(os.path.join(partition_data_folder, dataset, 'server', t, '')), exist_ok=True)
        # The server uses the train data of the clients training and all the test available
        if t == 'train':
            with open(os.path.join(partition_data_folder, dataset, 'server', t, 'data.json'), "w") as outfile:
                json.dump({'user_data': server_data}, outfile)
        else:
            with open(os.path.join(partition_data_folder, dataset, 'server', t, 'data.json'), "w") as outfile:
                json.dump({'user_data': all_data}, outfile)

    # Create a test dataset to test for backdoor attacks
    if target_class is not None:
        test_target = {}
        for user, data in all_data.items():
            # d[1] is y
            result = list(filter(lambda d: d[1] == target_class, zip(data['x'], data['y'])))
            test_target[user] = {}
            any(gather_test_target(test_target[user], d) for d in result)
        os.makedirs(os.path.join(partition_data_folder, dataset, 'server', 'target_test'), exist_ok=True)
        with open(os.path.join(partition_data_folder, dataset, 'server', 'target_test', 'data.json'), "w") as outfile:
            json.dump({'user_data': test_target}, outfile)

