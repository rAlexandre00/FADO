import json
import os


def split_data(all_data_folder, partition_data_folder, num_users):
    for t in ["train", 'test']:
        all_users = []
        all_data = {}
        data_files = os.listdir(os.path.join(all_data_folder, t))
        # Get all json files from all_data_folder
        data_files = [f for f in data_files if f.endswith(".json")]
        for f in data_files:
            with open(os.path.join(all_data_folder, t, f), 'r') as file:
                j = json.load(file)
            all_users.extend(j['users'][:num_users])
            all_data.update(j['user_data'])

        users = all_users[:num_users]

        for i, user_id in enumerate(users, start=1):
            os.makedirs(os.path.dirname(os.path.join(partition_data_folder, f'user_{i}', t, '')), exist_ok=True)
            # Create a dictionary with user ids and empty lists
            # new_user_data = {k: {'x': [], 'y': []} for k in new_users}
            # Set the corresponding user list
            user_data = {user_id: all_data[user_id]}
            with open(os.path.join(partition_data_folder, f'user_{i}', t, 'data.json'), "w") as outfile:
                json.dump({
                    'users': users,
                    'user_data': user_data
                }, outfile)

        # Create final file for server
        os.makedirs(os.path.dirname(os.path.join(partition_data_folder, f'server', t, '')), exist_ok=True)
        server_data = {user_id: all_data[user_id] for user_id in users}
        with open(os.path.join(partition_data_folder, 'server', t, 'data.json'), "w") as outfile:
            json.dump({
                'users': users,
                'user_data': server_data
            }, outfile)
