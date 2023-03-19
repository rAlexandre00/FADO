import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.ml.model.fado_module import FADOModule

fado_args = FADOArguments()

if fado_args.use_gpu:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        config = tf.config.experimental.set_memory_growth(device, True)


class NlaflDbpediaTf(FADOModule):

    def __init__(self):
        data_path = os.getenv("FADO_DATA_PATH", default='/app/data')
        embedding_matrix = np.load(os.path.join(data_path, 'train', 'dbpedia_embedding_matrix.npy'))
        self.model = build_model(embedding_matrix)

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


def build_model(embedding_matrix):
    """ Build the local model using pretrained glove embeddings
    Args:
        embedding_matrix

    Returns:
        object: model object
    """

    num_tokens = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]

    embedding_layer = tf.keras.layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(
            embedding_matrix),
        trainable=False,
    )

    int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)

    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(embedded_sequences)
    x = tf.keras.layers.Conv1D(
        128, 7, padding="valid", activation="relu", strides=3)(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    predictions = tf.keras.layers.Dense(
        14, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(int_sequences_input, predictions)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer="adam", metrics=["accuracy"])
    return model
