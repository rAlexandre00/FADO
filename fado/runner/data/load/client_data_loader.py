import logging

import numpy as np

from fado.builder.data.dataset import Dataset
from fado.runner.data.load.base_data_loader import DataLoader
from fado.cli.arguments.arguments import FADOArguments

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'client'})
fado_args = FADOArguments()


class ClientDataLoader(DataLoader):

    def __init__(self, data_dir, client_id) -> None:
        super().__init__(data_dir)
        self.client_id = client_id

    def read_data(self):
        train = {}

        # Load train data
        train_npz = np.load(self.train_path, allow_pickle=True)
        train['x'] = train_npz[f'{self.client_id}_x']
        train['y'] = train_npz[f'{self.client_id}_y'].tolist()

        return Dataset(train_data=train)
