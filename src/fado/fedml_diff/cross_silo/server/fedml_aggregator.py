import logging

import numpy as np

logger = logging.getLogger(__name__)


# class FedMLAggregator(object):

def data_silo_selection(self, round_idx, client_num_in_total, client_num_per_round):
    """This monkey patch is used to fix a bug in FedML where it assumed that every silo has the all dataset"""
    logging.info(
        "client_num_in_total = %d, client_num_per_round = %d" % (client_num_in_total, client_num_per_round)
    )
    assert client_num_in_total >= client_num_per_round

    return [i for i in range(client_num_per_round)]
