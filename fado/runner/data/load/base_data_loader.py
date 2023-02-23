from abc import ABC, abstractmethod
import os

import json
import logging

from fado.builder.data.dataset import Dataset
from fado.cli.arguments.arguments import FADOArguments

logger = logging.getLogger('fado')


class DataLoader(object):

    def __init__(self, data_dir) -> None:
        fado_args = FADOArguments()
        target_class_exists = 'target_class' in fado_args
        self.dataset = None

        target_test_path = None
        if target_class_exists:
            self.target_test_path = os.path.join(data_dir, "target_test", 'all_data.npz')

        self.train_path = os.path.join(data_dir, "train", "all_data.npz")
        self.test_path = os.path.join(data_dir, "test", "all_data.npz")

    @abstractmethod
    def read_data(self):
        pass


        # return Dataset(
        #     client_num,
        #     train_data_num,         # number samples in train
        #     test_data_num,          # number samples in test
        #     target_test_data_num,   # number of samples in target test
        #     class_num,              # number of classes
        #     train_data_global,      # batched train data
        #     test_data_global,       # batched test data
        #     target_test_global,     # batched target test data
        # )

