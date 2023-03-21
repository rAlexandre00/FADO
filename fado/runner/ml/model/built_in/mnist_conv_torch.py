import logging

import numpy as np
import os
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.ml.model.fado_module import FADOModule

fado_args = FADOArguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device - {device}')

logger = logging.LoggerAdapter(logging.getLogger("fado"), extra={'node_id': 'model'})


class MnistConvTorch(FADOModule):

    def __init__(self):
        self.model = build_model()

    def get_parameters(self):
        """ Retrieve the model's weights

        Returns:
            list: a list containing a whole state of the module
        """
        weights = [p.data for p in self.model.parameters()]
        return [w.cpu().numpy() for w in weights]

    def set_parameters(self, new_weights):
        """ Assign weights to the model

        Args:
            new_weights (list):
                a list containing a whole state of the module
        """
        i = 0
        for p in self.model.parameters():
            p.data = torch.from_numpy(new_weights[i])
            i += 1

    def _train_dataloader(self, x_train, y_train):
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=fado_args.batch_size, shuffle=True)

        optimizer = optim.SGD(self.model.parameters(), lr=fado_args.learning_rate)

        for epoch in range(fado_args.epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def train(self, x, y):
        """ Train the local model for one roung

        This can correspond to training for multiple epochs, or a single epoch.
        Returs return final weights, train loss, train accuracy

        Returns:
            tuple: final weights, train loss, train accuracy
        """
        self.model.to(device)
        self.model.train()

        x_train = torch.tensor(x).to(device)
        y_train = torch.tensor(y).to(device)

        self._train_dataloader(x_train, y_train)

        y_pred = self.model(x_train)
        # Calculate the cross-entropy loss
        loss = self.model.criterion(y_pred, y_train)

        # Calculate the accuracy
        _, predictions = torch.max(y_pred, dim=1)
        correct = (predictions == y_train).float()
        accuracy = torch.mean(correct)

        return self.get_parameters(), loss, accuracy

    def evaluate(self, x, y):
        # switch to evaluate mode
        self.model.eval()

        # Convert numpy arrays to PyTorch tensors
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).long()

        y_pred = self.model(x)
        # Calculate the cross-entropy loss
        loss = self.model.criterion(y_pred, y)

        # Calculate the accuracy
        _, predictions = torch.max(y_pred, dim=1)
        correct = (predictions == y).float()
        accuracy = torch.mean(correct)

        return loss.item(), accuracy.item()


def build_model():
    return MnistConv()


class MnistConv(nn.Module):
    def __init__(self):
        super(MnistConv, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 62)
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, x, noise=torch.Tensor()):
        x = x.reshape(-1, 1, 28, 28)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 32 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MnistConvLarge(nn.Module):
    def __init__(self):
        super(MnistConvLarge, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 62)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.SGD(self.model.parameters())

    def forward(self, x, noise=torch.Tensor()):
        x = x.reshape(-1, 1, 28, 28)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64 * 7 * 7)  # reshape Variable
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    _, predictions = torch.max(output, dim=1)
    correct = (predictions == target).float()
    acc = torch.mean(correct)

    return acc
