import logging
import os

import numpy as np

from fado.builder.data.shape.shaper import Shaper
from fado.cli.arguments.arguments import FADOArguments
from fado.constants import ALL_DATA_FOLDER

fado_args = FADOArguments()
DATA_FOLDER = os.path.join(ALL_DATA_FOLDER, fado_args.dataset)

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'builder'})


class LEAFShaper(Shaper):

    def shape(self):
        if fado_args.dataset == 'femnist':
            shape_femnist()
        else:
            raise Exception("NLAFL dataset not supported yet")


def shape_femnist():
    logger.info('Generating femnist dataset')
    trn_x, trn_y, tst_x, tst_y = load_femnist()

    partitioned_tst = partition_by_class(tst_x, tst_y)

    # Returns list of tuples, (data, labels)) for each client
    client_data = sample_data(trn_x, trn_y)

    test_target_x = partitioned_tst[fado_args.target_class]
    test_target_size = len(test_target_x)
    test_target_x_server = test_target_x[:test_target_size//2]
    test_target_x_attacker = test_target_x[test_target_size//2:]

    os.makedirs(os.path.join(DATA_FOLDER, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'test'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'target_test'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'target_test_attacker'), exist_ok=True)
    np.savez_compressed(os.path.join(DATA_FOLDER, 'train', 'all_data'), **client_data)
    np.savez_compressed(os.path.join(DATA_FOLDER, 'test', 'all_data'), x=tst_x, y=tst_y)
    np.savez_compressed(os.path.join(DATA_FOLDER, 'target_test', 'all_data'),
                        x=test_target_x_server,
                        y=len(test_target_x_server)*[fado_args.target_class])
    np.savez_compressed(os.path.join(DATA_FOLDER, 'target_test_attacker', 'all_data'),
                        x=test_target_x_attacker,
                        y=len(test_target_x_attacker)*[fado_args.target_class])


def load_femnist():
    """ Load the FEMNIST dataet (LEAF)
    Returns:
        tuple: tuple of numpy arrays trn_x, trn_y, tst_x, tst_y
    """

    trn_x = np.load(os.path.join(DATA_FOLDER, 'trn_x_femnist.npz'), allow_pickle=True)['data']
    trn_y = np.load(os.path.join(DATA_FOLDER, 'trn_y_femnist.npz'), allow_pickle=True)['data']
    tst_x = np.load(os.path.join(DATA_FOLDER, 'tst_x_femnist.npz'), allow_pickle=True)['data']
    tst_y = np.load(os.path.join(DATA_FOLDER, 'tst_y_femnist.npz'), allow_pickle=True)['data']
    return trn_x, trn_y, tst_x, tst_y

def partition_by_class(x, y):
    """ Given a dataset matrix and labels, return the data matrix partitioned by class.
    The list of classes is assumed to be the number of classes for the dataset.
    Example output:
        [ [class 1's x ..], [class 2's x ..] ,  ... [class 10^s x ..]  ]
    Args:
        x (numpy.ndarray): data matrix
        y (numpy.ndarray): data labels
    Returns:
        list: Partitioned data matrix, as list of ndarray objects
    """

    all_x = []
    y_list = range(fado_args.num_classes)

    for y_val in y_list:
        all_x.append(x[np.where(y == y_val)[0]])
    return all_x


def sample_leaf(
        all_x,
        all_y,
        num_clients
):
    """
    Args:
        all_x (list): data matrix, as list of ndarray objects
        all_y (list): labels, as list of integers
        num_clients (int): number of clients
    Returns:
        dict: with keys x_i and y_i being i the client id
    """

    # Initialize per-client data structures
    clients = {}

    np.random.seed(fado_args.random_seed)
        
    idx = np.random.permutation(len(all_x))

    all_x_shuffled = all_x[idx]
    all_y_shuffled = all_y[idx]
    
    for i in range(num_clients):
        clients[f'{i+1}_x'] = all_x_shuffled[i]
        clients[f'{i+1}_y'] = all_y_shuffled[i]

    return clients

def sample_data(x, y):
    client_data = sample_leaf(
        x,
        y,
        fado_args.number_clients
    )
    return client_data