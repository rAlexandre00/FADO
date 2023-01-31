from abc import ABC
from abc import abstractmethod
import os

import json
import logging
import numpy as np

import torch


cwd = os.getcwd()

__all__ = ['DataLoader']

logger = logging.getLogger('fado')

class DataLoader:

    def __init__(self, args) -> None:
        self.args = args
        self.dataset = None

        if self.args is None:
            raise Exception("args is not defined!")
        
        if hasattr(args, 'target_class'):
            target_test_path = args.data_cache_dir + "/target_test"
        else:
            target_test_path = None

        # read data
        train_data, test_data, self.target_test_data = self.read_data(
            self.args.data_cache_dir + "/train",
            self.args.data_cache_dir + "/test",
            target_test_data_dir=target_test_path
        )

        # process data

        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            target_test_data,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = self.process_data(train_data, test_data, self.target_test_data)

        # build dataset from data

        self.args.client_num_in_total = client_num
        self.dataset = [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num
        ]

    def process_data(self, train_data, test_data, target_test_data):
        """Creates the parameters needed for a dataset out of a group of train and test files

            Assumes:
                the data in the input directories are .json files with pairs ('user_id','user_data')
                the set of train set users is the same as the set of test set users

            Parameters:
                args: FedML arguments
                train_path (str): folder with the train data in the form of json files
                test_path (str): folder with the test data in the form of json files
                target_test_path (str): Optional

            Returns:
                Properties that form a dataset
                (client_num, train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict,
                train_data_local_dict, test_data_local_dict, class_num)

        """

        train_data_num = 0
        test_data_num = 0
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        train_data_local_num_dict = dict()
        train_data_global = list()
        test_data_global = list()
        target_test_global = list()
        client_idx = 0

        for user in train_data.keys():
            if train_data[user]:
                user_train_data_num = len(train_data[user]["x"])
                user_test_data_num = len(test_data[user]["x"])
                train_data_num += user_train_data_num
                test_data_num += user_test_data_num
                train_data_local_num_dict[client_idx] = user_train_data_num

                # transform to batches
                train_batch = self.batch_data(train_data[user], self.args.batch_size)
                test_batch = self.batch_data(test_data[user], self.args.batch_size)

                # index using client index
                train_data_local_dict[client_idx] = train_batch
                test_data_local_dict[client_idx] = test_batch
                train_data_global += train_batch
                test_data_global += test_batch
            client_idx += 1

        for user in target_test_data.keys():
            if target_test_data[user]:
                target_test_batch = self.batch_data(target_test_data[user], self.args.batch_size)
                target_test_global += target_test_batch

        client_num = client_idx
        # Must be read from args (!!!)
        class_num = 62

        del train_data
        del test_data
        del target_test_data

        return (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            target_test_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        )

    def read_data(self, train_data_dir, test_data_dir, target_test_data_dir=None):
        """parses data in given train and test data directories

        assumes:
        - the data in the input directories are .json files with
            key 'user_data'
        - the set of train set users is the same as the set of test set users

        Return:
            clients: list of non-unique client ids
            train_data: dictionary of train data
            test_data: dictionary of test data
        """
        train_data = {}
        test_data = {}
        target_test_data = {}
        for data_dir, data_dict in zip([train_data_dir, test_data_dir, target_test_data_dir],
                                    [train_data, test_data, target_test_data]):
            if data_dir is None:  # target_test_data_dir is optional
                continue
            data_files = os.listdir(data_dir)
            data_files = [f for f in data_files if f.endswith(".json")]
            for f in data_files:
                file_path = os.path.join(data_dir, f)
                with open(file_path, "r") as inf:
                    cdata = json.load(inf)
                data_dict.update(cdata["user_data"])

        return train_data, test_data, target_test_data

    def batch_data(self, data, batch_size):
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
            batched_x, batched_y = self.convert_to_tensor(batched_x, batched_y)
            batch_data.append((batched_x, batched_y))
        return batch_data

    def convert_to_tensor(self, batched_x, batched_y):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float().reshape(-1, 28, 28)
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()

        return batched_x, batched_y
    