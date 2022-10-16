import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from fedml.core.security.common.utils import (
    get_malicious_client_id_list,
    replace_original_class_with_target_class,
    log_client_data_statistics,
)

from .attack_base import DataAttack

"""
ref: Tolpegin, Vale, Truex,  "Data Poisoning Attacks Against Federated Learning Systems."  (2021).
attack @client, added by Yuhui, 07/08/2022
"""


class LabelFlippingAttack(DataAttack):
    def __init__(
        self, args
    ):

        super().__init__()
        self.original_class_list = args.original_class_list
        self.target_class_list = args.target_class_list
        self.batch_size = args.batch_size

    def attack_data(self, dataset):
        tmp_local_dataset_X = torch.Tensor([])
        tmp_local_dataset_Y = torch.Tensor([])
        for (data, target) in dataset:
            tmp_local_dataset_X = torch.cat((tmp_local_dataset_X, data))
            tmp_local_dataset_Y = torch.cat((tmp_local_dataset_Y, target)).to(torch.long)
        tmp_Y = replace_original_class_with_target_class(
            data_labels=tmp_local_dataset_Y,
            original_class_list=self.original_class_list,
            target_class_list=self.target_class_list,
        )
        dataset = TensorDataset(tmp_local_dataset_X, tmp_Y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        return data_loader
        #log_client_data_statistics(self.poisoned_client_list, poisoned_dataset)
