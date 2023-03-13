import itertools
import os

import numpy as np
import pandas as pd

from fado.cli.arguments.arguments import FADOArguments
from fado.constants import RESULTS_DIRECTORY

fado_args = FADOArguments()


def generate_table(r):
    """

    :param r: Round
    :return:
    """
    # Find result file static part
    start_file_str = fado_args.results_file_name[:fado_args.results_file_name.find('{')]
    # Find all file names starting with that string
    npy_files = [r for r in os.listdir(RESULTS_DIRECTORY) if r.startswith(start_file_str)]

    results = {k: [] for k in fado_args.vary.keys()}
    results['target_accuracy'] = []

    for npy_file in npy_files:
        data = np.load(os.path.join(RESULTS_DIRECTORY, npy_file), allow_pickle=True).item()
        for key in fado_args.vary.keys():
            results[key].append(data['fado_args'].get_argument(key))
        results['target_accuracy'].append(np.mean(data['per_round_target_accuracy'][r-1:r+4]))

    df = pd.DataFrame(results)
    fado_args.vary.pop('random_seed')
    p_table = pd.pivot_table(df, values='target_accuracy', index=list(fado_args.vary.keys()), aggfunc=np.mean)

    print(p_table)
