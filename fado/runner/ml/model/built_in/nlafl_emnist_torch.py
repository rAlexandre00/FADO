import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.ml.model.fado_module import FADOModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device - {device}')

fado_args = FADOArguments()


class NlaflEmnistTorch(FADOModule):

    def __init__(self):
        self.model = NeuralNet()
        
    def get_parameters(self):
        """ Retrieve the model's weights

        Returns:
            list: a list containing a whole state of the module
        """
        weights = [p.data for p in self.model.parameters()]
        return np.array([w.cpu().numpy() for w in weights], dtype=object)

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

        if self.model.momentum > 0.0:
            optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=self.model.momentum)
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        for epoch in range(fado_args.epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def train(self, x, y):
        """ Train the local model for one round

        This can correspond to training for multiple epochs, or a single epoch.
        Returns final weights, train loss, train accuracy

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
        return self.model.evaluate(x, y)


class NeuralNet(nn.Module):
    def __init__(self, momentum=0.0, dropouts=False):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.dropouts = dropouts
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 62)
        self.momentum = momentum
        self.criterion = nn.CrossEntropyLoss().to(device)

    def forward(self, x):
        # Change to channel last
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        if self.dropouts:
            x = F.dropout(x, 0.25)
        x = x.reshape(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        if self.dropouts:
            x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return x

    def evaluate(self, x, y):
        self.eval()
        # Convert numpy arrays to PyTorch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        y_pred = self(x)
        y = y.long()
        _, predictions = torch.max(y_pred, dim=1)
        # Calculate the cross-entropy loss
        loss = self.criterion(y_pred, y)

        # Calculate the accuracy
        _, predictions = torch.max(y_pred, dim=1)
        correct = (predictions == y).float()
        accuracy = torch.mean(correct)

        return loss.item(), accuracy.item()
