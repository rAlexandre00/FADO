import logging
import os
import shutil

import numpy as np

from fado.builder.data.shape.shaper import Shaper
from fado.cli.arguments.arguments import FADOArguments
from fado.constants import ALL_DATA_FOLDER

ALPHA = 1

fado_args = FADOArguments()
DATA_FOLDER = os.path.join(ALL_DATA_FOLDER, fado_args.dataset)

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'builder'})


class NLAFLShaper(Shaper):

    def __init__(self):
        self.num_users = None
        self.client_size = None
        self.target_fraction = None
        self.poison_count = fado_args.poison_count_multiplier * (fado_args.num_pop_clients // 3)

    def shape(self):
        if fado_args.dataset == 'nlafl_emnist':
            trn_x, trn_y, tst_x, tst_y = load_emnist()
            logger.info('Generating nlafl emnist dataset')
            self.num_users = fado_args.number_clients
            self.client_size = 1000
            self.target_fraction = 0.5
            self.shape_data(trn_x, trn_y, tst_x, tst_y)
        elif fado_args.dataset == 'nlafl_fashionmnist':
            trn_x, trn_y, tst_x, tst_y = load_fashionmnist()
            logger.info('Generating nlafl fashionmnist dataset')
            self.num_users = fado_args.number_clients
            self.client_size = 400
            self.target_fraction = 0.6
            self.shape_data(trn_x, trn_y, tst_x, tst_y)
        elif fado_args.dataset == 'nlafl_dbpedia':
            trn_x, trn_y, tst_x, tst_y = load_dbpedia()
            logger.info('Generating nlafl dbpedia dataset')
            self.num_users = fado_args.number_clients
            self.client_size = 1000
            self.target_fraction = 0.6
            self.shape_data(trn_x, trn_y, tst_x, tst_y)
            shutil.copy(os.path.join(DATA_FOLDER, 'dbpedia_embedding_matrix.npy'), os.path.join(DATA_FOLDER, 'train'))
        else:
            raise Exception("NLAFL dataset not supported yet")

    def shape_data(self, trn_x, trn_y, tst_x, tst_y):
        partitioned_trn = partition_by_class(trn_x, trn_y)
        partitioned_tst = partition_by_class(tst_x, tst_y)
        trn_y, tst_y = np.eye(fado_args.num_classes)[trn_y], np.eye(fado_args.num_classes)[tst_y]
        # Sample data from the original dataset according to a Dirichlet distribution.
        # Returns list of tuples, (data, labels)) for each client
        client_data = self.sample(partitioned_trn)
        write_files(client_data, tst_x, tst_y, partitioned_tst)

    def sample(self, partitioned):
        if self.poison_count > 0:
            client_data = fixed_poison(
                partitioned,
                self.num_users,
                self.client_size,
                self.poison_count,
                targ_class=fado_args.target_class,
                client_targ=fado_args.num_pop_clients,
                targ_frac=self.target_fraction,
                alpha=ALPHA,
            )

        else:
            client_data = fixed_sample(
                partitioned,
                self.num_users,
                self.client_size,
                targ_class=fado_args.target_class,
                client_targ=fado_args.num_pop_clients,
                targ_frac=self.target_fraction,
                alpha=ALPHA,
            )
        return client_data


def write_files(client_data, tst_x, tst_y, partitioned_tst):
    test_target_x = partitioned_tst[fado_args.target_class]
    test_target_size = len(test_target_x)
    test_target_x_attacker = test_target_x[:test_target_size // 2]
    test_target_x_server = test_target_x[test_target_size // 2:]

    os.makedirs(os.path.join(DATA_FOLDER, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'test'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'target_test'), exist_ok=True)
    os.makedirs(os.path.join(DATA_FOLDER, 'target_test_attacker'), exist_ok=True)
    np.savez_compressed(os.path.join(DATA_FOLDER, 'train', 'all_data'), **client_data)
    np.savez_compressed(os.path.join(DATA_FOLDER, 'test', 'all_data'), x=tst_x, y=tst_y)
    np.savez_compressed(os.path.join(DATA_FOLDER, 'target_test', 'all_data'),
                        x=test_target_x_server,
                        y=np.eye(fado_args.num_classes)[len(test_target_x_server) * [fado_args.target_class]])
    np.savez_compressed(os.path.join(DATA_FOLDER, 'target_test_attacker', 'all_data'),
                        x=test_target_x_attacker,
                        y=np.eye(fado_args.num_classes)[len(test_target_x_attacker) * [fado_args.target_class]])


def load_emnist():
    """ Load the EMNIST dataet
    Returns:
        tuple: tuple of numpy arrays trn_x, trn_y, tst_x, tst_y
    """

    trn_x = np.load(os.path.join(DATA_FOLDER, 'trn_x_emnist.npy'))
    trn_y = np.load(os.path.join(DATA_FOLDER, 'trn_y_emnist.npy'))
    tst_x = np.load(os.path.join(DATA_FOLDER, 'tst_x_emnist.npy'))
    tst_y = np.load(os.path.join(DATA_FOLDER, 'tst_y_emnist.npy'))
    return trn_x, trn_y, tst_x, tst_y


def load_fashionmnist():
    """ Load the EMNIST dataet
    Returns:
        tuple: tuple of numpy arrays trn_x, trn_y, tst_x, tst_y
    """

    trn_x = np.load(os.path.join(DATA_FOLDER, 'trn_x_fashionMnist.npy'))
    trn_y = np.load(os.path.join(DATA_FOLDER, 'trn_y_fashionMnist.npy'))
    tst_x = np.load(os.path.join(DATA_FOLDER, 'tst_x_fashionMnist.npy'))
    tst_y = np.load(os.path.join(DATA_FOLDER, 'tst_y_fashionMnist.npy'))
    return trn_x, trn_y, tst_x, tst_y


def load_dbpedia():
    """ Load the EMNIST dataet
    Returns:
        tuple: tuple of numpy arrays trn_x, trn_y, tst_x, tst_y
    """

    trn_x = np.load(os.path.join(DATA_FOLDER, 'trn_x_dbpedia.npy'))
    trn_y = np.load(os.path.join(DATA_FOLDER, 'trn_y_dbpedia.npy'))
    tst_x = np.load(os.path.join(DATA_FOLDER, 'tst_x_dbpedia.npy'))
    tst_y = np.load(os.path.join(DATA_FOLDER, 'tst_y_dbpedia.npy'))
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


def fixed_sample(
        all_x,
        num_clients,
        client_size,
        targ_class=0,
        client_targ=5,
        targ_frac=.2,
        alpha=100
):
    """ Use a Dirichlet distribution to assign target class samples to clients
    `all_x` -> [ [class 1's x ..], [class 2's x ..] ,  ... [class 10^s x ..]  ]
    `client Size` is used to calculate number samples for each class with
    dirichlet distirbution alpha
    Args:
        all_x (list): partitioned data matrix, as list of ndarray objects
        num_clients (int): number of clients
        client_size (int): desired number of samples per client
        targ_class (int, optional): identifier of target class. Defaults to 0
        client_targ (int, optional): number of clients having target class points. Defaults to 5
        targ_frac (float, optional): fraction of target class points for clients having them. Defaults to .2
        alpha (int, optional): Dirichlet parameter alpha. Defaults to 100
    Returns:
        dict: with keys x_i and y_i being i the client id
    """

    num_classes = fado_args.num_classes
    num_nontarget = num_classes - 1

    # Initialize per-client data structures
    clients = {}
    orig_dirichlets = np.random.dirichlet([alpha] * num_nontarget, num_clients)
    all_dirichlets = np.zeros((num_clients, num_classes))

    # Fill up the columns of `all_dirichlets` up to the target class,
    # and from the one following the target class to the end using the
    # values generated in `orig_dirichlets`
    all_dirichlets[:, :targ_class] = orig_dirichlets[:, :targ_class]
    all_dirichlets[:, targ_class + 1:] = orig_dirichlets[:, targ_class:]

    # targ_x is the numpy array of all target class samples
    targ_x = all_x[targ_class]

    for i in range(num_clients):
        this_x, this_y = [], []
        total_ct = client_size

        # The first client_targ clients will have the target class samples
        if i < client_targ:
            # number of target class samples for client i
            num_targ = int(total_ct * targ_frac)
            total_ct -= num_targ

            # Assign the target class samples to client i and create a label vector
            this_x.append(targ_x[:num_targ])
            this_y.append(np.zeros(num_targ, dtype=int) + targ_class)

            # Remove the samples used for this client from targ_x
            targ_x = targ_x[num_targ:]

        counts = (total_ct * all_dirichlets[i]).astype(int)
        assert counts[targ_class] == 0

        for y in range(num_classes):
            # Ignore the target class
            if y == targ_class:
                continue

            y_ct = counts[y].astype(int)
            this_x.append(all_x[y][:y_ct])
            all_x[y] = all_x[y][y_ct:]
            this_y.append(np.zeros(y_ct, dtype=int) + y)

        this_x = np.concatenate(this_x)
        this_y = np.eye(fado_args.num_classes)[np.concatenate(this_y)]
        assert this_x.shape[0] == this_y.shape[0]
        clients[f'{i + 1}_x'] = this_x
        clients[f'{i + 1}_y'] = this_y

    return clients


def fixed_poison(
        all_x,
        num_clients,
        client_size,
        poison_ct,
        targ_class=0,
        client_targ=5,
        targ_frac=.2,
        alpha=100
):
    """
    Args:
        all_x (list): partitioned data matrix, as list of ndarray objects
        num_clients (int): number of clients
        client_size (int): desired number of samples per client
        poison_ct (int): number of clients participating in the poisoning attack
        targ_class (int, optional): identifier of target class. Defaults to 0
        client_targ (int, optional): number of clients having target class points. Defaults to 5
        targ_frac (float, optional): fraction of target class points for clients having them. Defaults to .2
        alpha (int, optional): Dirichlet parameter alpha. Defaults to 100
        seed (int, optional): seed for PRNGs. Defaults to None
    Returns:
        dict: with keys x_i and y_i being i the client id
    """

    num_classes = fado_args.num_classes
    num_nontarget = num_classes - 1

    # Initialize per-client data structures
    clients = {}
    orig_dirichlets = np.random.dirichlet([alpha] * num_nontarget, num_clients)
    all_dirichlets = np.zeros((num_clients, num_classes))

    # Fill up the columns of `all_dirichlets` up to the target class,
    # and from the one following the target class to the end using the
    # values generated in `orig_dirichlets`
    all_dirichlets[:, :targ_class] = orig_dirichlets[:, :targ_class]
    all_dirichlets[:, targ_class + 1:] = orig_dirichlets[:, targ_class:]

    # targ_x is the numpy array of all target class samples
    targ_x = all_x[targ_class]

    for i in range(num_clients):
        this_x, this_y = [], []
        total_ct = client_size

        # The first client_targ clients will have the target class samples
        if i < client_targ:
            # number of target class samples for client i
            num_targ = int(total_ct * targ_frac)
            total_ct -= num_targ

            # Assign the target class samples to client i and create a label vector
            this_x.append(targ_x[:num_targ])
            this_y.append(np.zeros(num_targ, dtype=np.int) + targ_class)

            # Remove the samples used for this client from targ_x
            targ_x = targ_x[num_targ:]

        # The successive `poison_ct` clients will have the poisoned points
        elif i < client_targ + poison_ct:
            num_targ = int(total_ct * targ_frac)
            total_ct -= num_targ
            counts = (total_ct * all_dirichlets[i]).astype(np.int)

            # Flip the labels for the target class samples
            for y in range(num_classes):
                if y == targ_class:
                    y_ct = num_targ
                    y_local = (y + 1) % num_classes

                else:
                    y_ct = counts[y].astype(np.int)
                    y_local = y

                # Assign the samples to this client
                this_x.append(all_x[y][:y_ct])
                this_y.append(np.zeros(y_ct, dtype=np.int) + y_local)

                # Remove the samples used for this client
                all_x[y] = all_x[y][y_ct:]

            this_x = np.concatenate(this_x)
            this_y = np.eye(fado_args.num_classes)[np.concatenate(this_y)]
            assert this_x.shape[0] == this_y.shape[0]
            clients[f'{i + 1}_x'] = this_x
            clients[f'{i + 1}_y'] = this_y
            continue

        counts = (total_ct * all_dirichlets[i]).astype(np.int)
        assert counts[targ_class] == 0

        for y in range(num_classes):
            # Ignore the target class
            if y == targ_class:
                continue

            y_ct = counts[y].astype(np.int)
            this_x.append(all_x[y][:y_ct])
            all_x[y] = all_x[y][y_ct:]
            this_y.append(np.zeros(y_ct, dtype=np.int) + y)

        this_x = np.concatenate(this_x)
        this_y = np.eye(fado_args.num_classes)[np.concatenate(this_y)]
        assert this_x.shape[0] == this_y.shape[0]
        clients[f'{i + 1}_x'] = this_x
        clients[f'{i + 1}_y'] = this_y

    return clients
