import json
import logging
import os


__all__ = ['split_data']


def split_data(all_data_folder, partition_data_folder, num_users):
    """Takes a folder with data json files and created 'num_users' folder each with specific client data

        Assumes:
            the data in the .json files have pairs ('user_id','user_data')

        Parameters:
            all_data_folder: folder with two folder, train and test data each with data json files
            partition_data_folder: output folder for client partition json files
            num_users: number of users to generate partitions for

    """
    for t in ["train", 'test']:
        all_data = {}
        data_files = os.listdir(os.path.join(all_data_folder, t))
        # Get all json files from all_data_folder
        data_files = [f for f in data_files if f.endswith(".json")]
        for f in data_files:
            with open(os.path.join(all_data_folder, t, f), 'r') as file:
                j = json.load(file)
            all_data.update(j['user_data'])

        users = list(all_data.keys())
        users.sort()
        users = users[:num_users]

        for i, user_id in enumerate(users, start=1):
            os.makedirs(os.path.dirname(os.path.join(partition_data_folder, f'user_{i}', t, '')), exist_ok=True)
            # Create a dictionary with user ids and empty lists
            user_data = {k: {} for k in users}
            # Set the corresponding user list
            user_data[user_id] = all_data[user_id]
            with open(os.path.join(partition_data_folder, f'user_{i}', t, 'data.json'), "w") as outfile:
                json.dump({'user_data': user_data}, outfile)

        # The server uses the all_data_folder
