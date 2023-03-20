import logging

import numpy as np

from fado.builder.data.dataset import Dataset
from fado.runner.data.load.base_data_loader import DataLoader
from fado.cli.arguments.arguments import FADOArguments

logger = logging.getLogger('fado')
fado_args = FADOArguments()


class ServerDataLoader(DataLoader):

    def __init__(self, data_dir) -> None:
        super().__init__(data_dir)

    def read_data(self):
        train = {'x': [], 'y': []}
        test = {}
        target_test = {}

        # Load train data
        train_npz = np.load(self.train_path, allow_pickle=True)
        for client_id in range(fado_args.number_clients):
            train['x'].append(train_npz[f'{client_id+1}_x'])
            train['y'].append(train_npz[f'{client_id+1}_y'])

        # Load test data
        test_npz = np.load(self.test_path, allow_pickle=True)
        test['x'], test['y'] = test_npz['x'], test_npz['y']

        # Load target test data
        if self.target_test_path:
            target_test_npz = np.load(self.target_test_path, allow_pickle=True)
            target_test['x'], target_test['y'] = target_test_npz['x'], target_test_npz['y']
            return Dataset(train_data=train, test_data=test, target_test_data=target_test)
        else:
            return Dataset(train_data=train, test_data=test)
