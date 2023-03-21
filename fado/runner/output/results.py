import os.path

import numpy as np

from fado.cli.arguments.arguments import FADOArguments


class Results:

    def __init__(self):
        self.results_dict = {}

    def add_round(self, key, value):
        self.results_dict.setdefault(key, [])
        self.results_dict[key].append(value)

    def write_to_file(self):
        fado_args = FADOArguments()
        self.results_dict['fado_args'] = fado_args
        results_file_name = fado_args.results_file_name.format(**fado_args.__dict__)
        results_folder_path = os.getenv("RESULTS_FILE_PATH", default="/app/results/")
        os.makedirs(results_folder_path, exist_ok=True)
        output_path = os.path.join(results_folder_path, results_file_name)
        np.save(output_path, self.results_dict)

    def load_from_file(self, file_path):
        self.results_dict = np.load(file_path, allow_pickle=True)

