import json
import os

import numpy as np
from fedml.ml.engine import ml_engine_adapter

cwd = os.getcwd()

import logging


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    train_data = {}
    test_data = {}

    for i, data_dir, data_dict in zip([0, 1], [train_data_dir, test_data_dir], [train_data, test_data]):
        data_files = os.listdir(data_dir)
        data_files = [f for f in data_files if f.endswith(".json")]
        for f in data_files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, "r") as inf:
                cdata = json.load(inf)
            data_dict.update(cdata["user_data"])
            if i == 0:
                clients.extend(cdata["users"])

    return clients, train_data, test_data


def batch_data(args, data, batch_size):
    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i: i + batch_size]
        batched_y = data_y[i: i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data(
        args, batch_size, train_path=os.path.join(os.getcwd(), "data", "train"),
        test_path=os.path.join(os.getcwd(), "data", "test")
):
    users, train_data, test_data = read_data(train_path, test_path)

    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0
    logging.info("loading data...")
    for u in users:
        if u in train_data:
            user_train_data_num = len(train_data[u]["x"])
            user_test_data_num = len(test_data[u]["x"])
            train_data_num += user_train_data_num
            test_data_num += user_test_data_num
            train_data_local_num_dict[client_idx] = user_train_data_num

            # transform to batches
            train_batch = batch_data(args, train_data[u], batch_size)
            test_batch = batch_data(args, test_data[u], batch_size)

            # index using client index
            train_data_local_dict[client_idx] = train_batch
            test_data_local_dict[client_idx] = test_batch
            train_data_global += train_batch
            test_data_global += test_batch
        client_idx += 1
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 10

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
