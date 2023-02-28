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

        if target_class_exists:
            self.target_test_path = os.path.join(data_dir, "target_test", 'all_data.npz')
            self.target_test_attacker_path = os.path.join(data_dir, "target_test_attacker", "all_data.npz")

        self.train_path = os.path.join(data_dir, "train", "all_data.npz")
        self.test_path = os.path.join(data_dir, "test", "all_data.npz")

    @abstractmethod
    def read_data(self):
        pass
