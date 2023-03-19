import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.ml.model.fado_module import FADOModule

fado_args = FADOArguments()

if fado_args.use_gpu:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        config = tf.config.experimental.set_memory_growth(device, True)


class NlaflFashionmnistTf(FADOModule):

    def __init__(self):
        self.model = build_model()

    def get_parameters(self):
        """ Retrieve the model's weights

        Returns:
            list: a list containing a whole state of the module
        """
        return np.array(self.model.get_weights(), dtype=object)

    def set_parameters(self, new_weights):
        """ Assign weights to the model

        Args:
            new_weights (list):
                a list containing a whole state of the module
        """
        self.model.set_weights(new_weights)

    def train(self, x, y):
        """ Train the local model for one round

        This can correspond to training for multiple epochs, or a single epoch.
        Returns final weights, train loss, train accuracy

        Returns:
            tuple: final weights, train loss, train accuracy
        """

        self.model.fit(
            x,
            y,
            epochs=fado_args.epochs,
            batch_size=fado_args.batch_size,
            # verbose=1
            verbose=0,
            use_multiprocessing=True
        )

        score = self.model.evaluate(x, y, verbose=0, use_multiprocessing=True, workers=10)

        return self.model.get_weights(), score[0], score[1]

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0, use_multiprocessing=True, workers=10)


def build_model(momentum=0.0, dropouts=False):
    """ Build the local model
    Args:
        momentum (float, optional): momentum value for SGD. Defaults to 0.0.
        dropouts (bool, optional): if True use dropouts. Defaults to False.
    Returns:
        Sequential: model object
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropouts:
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    if momentum:
        sgd = SGD(learning_rate=0.1, momentum=momentum)
    else:
        sgd = SGD(learning_rate=0.1)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=sgd, metrics=['accuracy'])

    return model
