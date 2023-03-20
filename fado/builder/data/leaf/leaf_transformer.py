import os, sys, json

import numpy as np

def leaf_transformer(path):

    assert type(path) == str

    for t in ["train", 'test']:

        x = []
        y = []

        all_data = {}

        # get from argv...
        data_files = os.listdir(os.path.join(path, t))
        # Get all json files from all_data_folder
        data_files = [f for f in data_files if f.endswith(".json")]
        for f in data_files:
            with open(os.path.join(path, t, f), 'r') as file:
                j = json.load(file)
            all_data.update(j['user_data'])

        users = list(all_data.keys())

        for _, user_id in enumerate(users, start=1):

            user_x = np.array(all_data[user_id]['x'], dtype=np.float32)
            user_y = np.array(all_data[user_id]['y'])
            x.append(user_x.reshape((len(user_x), 28, 28, 1)))
            y.append(user_y)

        try:
            x = np.array(x, dtype=np.ndarray) if t == 'train' else np.concatenate(x, axis=0)
            y = np.array(y, dtype=np.ndarray) if t == 'train' else np.concatenate(y, axis=0)
        except ValueError as e:
            print(y[0])
        np.savez_compressed(os.path.join(path, f'{"trn" if t == "train" else "tst"}_x_femnist'), data=x)
        np.savez_compressed(os.path.join(path, f'{"trn" if t == "train" else "tst"}_y_femnist'), data=y)

        print(f'finished {t}')

    
            

if __name__ == '__main__':
    # main func

    leaf_transformer(sys.argv[1])