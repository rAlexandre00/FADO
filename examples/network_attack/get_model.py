import logging

import torch
from fedml.model.cv.cnn import CNN_DropOut
from torch import nn

logger = logging.getLogger('fado')


# def build_model(momentum=0.0, dropouts=False):
#     """ Build the local model
#     Args:
#         momentum (float, optional): momentum value for SGD. Defaults to 0.0.
#         dropouts (bool, optional): if True use dropouts. Defaults to False.
#     Returns:
#         object: model object
#     """
#
#     model = Sequential()
#     model.add(Conv2D(32, kernel_size=(3, 3),
#                      activation='relu', input_shape=(28, 28, 1)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     if dropouts:
#         model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     if dropouts:
#         model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#
#     if momentum:
#         sgd = SGD(lr=0.1, momentum=momentum)
#     else:
#         sgd = SGD(lr=0.1)
#
#     model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                   optimizer=sgd, metrics=['accuracy'])
#
#     return model

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.hidden = nn.Linear(64*12*12, 128) # TODO: INPUT?
        self.act = nn.ReLU()
        self.out = nn.Linear(128, 62)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        # spatial size: (28,28)    # number of channels = 1
        x = self.act(self.conv1(x))     # Conv2D
        # spatial size: (26,26)    # number of channels = 32
        x = self.act(self.conv2(x))     # Conv2D
        # spatial size: (24,24)    # number of channels = 64
        x = self.pool(x)                # MaxPooling2D
        # spatial size: (12,12)    # number of channels = 64
        x = self.drop1(x)               # TODO: Dropout?

        # spatial size: (12,12)    # number of channels = 64
        x = x.view(x.size(0), -1)       # Flatten
        x = self.act(self.hidden(x))    # Dense
        x = self.drop2(x)               # TODO: Dropout?

        x = self.out(x)                 # Dense
        return x

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

def get_model():
    # return CNN_DropOut(False)
    return NeuralNet()