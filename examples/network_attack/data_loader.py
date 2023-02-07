import numpy as np
import torch

from fado.data.data_loader import DataLoader


class HelloDataLoader(DataLoader):

    def __init__(self, args) -> None:
        super().__init__(args)

    def convert_to_tensor(self, batched_x, batched_y):
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()

        return batched_x, batched_y
    
def get_data_loader():
    return HelloDataLoader
    