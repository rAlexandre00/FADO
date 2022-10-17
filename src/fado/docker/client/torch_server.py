import os

import fedml
import torch

from fedml import FedMLRunner
from fado.data.data_loader import load_partition_data

from get_model import get_model


def load_data(args):
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data(
        args,
        args.batch_size,
        train_path=args.data_cache_dir + "/train",
        test_path=args.data_cache_dir + "/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # load model (the size of MNIST image is 28 x 28)
    model = get_model()

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
