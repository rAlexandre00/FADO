import logging

import numpy as np

logger = logging.getLogger(__name__)


# class FedMLAggregator(object):

def data_silo_selection(self, round_idx, client_num_in_total, client_num_per_round):
    """
    Args:
        round_idx: round index, starting from 0
        client_num_in_total: this is equal to the users in a synthetic data,
                                e.g., in synthetic_1_1, this value is 30
        client_num_per_round: the number of edge devices that can train
    Returns:
        data_silo_index_list: e.g., when client_num_in_total = 30, client_num_in_total = 3,
                                    this value is the form of [0, 11, 20]
    """
    logging.info(
        "client_num_in_total = %d, client_num_per_round = %d" % (client_num_in_total, client_num_per_round)
    )
    assert client_num_in_total >= client_num_per_round

    return [i for i in range(client_num_per_round)]
