import logging

import numpy as np

from fado.builder.data.dataset import Dataset
from fado.runner.data.load.base_data_loader import DataLoader
from fado.cli.arguments.arguments import FADOArguments

logger = logging.getLogger('fado')
fado_args = FADOArguments()


class AttackerDataLoader(DataLoader):

    def __init__(self, data_dir) -> None:
        super().__init__(data_dir)

    def read_data(self):
        test = {}

        # Load test data
        test_npz = np.load(self.target_test_attacker_path, allow_pickle=True)
        test['x'], test['y'] = test_npz['x'], test_npz['y']

        return Dataset(test_data=test)
