from fado.cli.arguments.arguments import FADOArguments


class Results:

    def __init__(self):
        args = FADOArguments()
        self.output_path = args.results_file_name.format(**args.__dict__)
        self.results_dict = {}

    def set_constant(self, key, value):
        self.results_dict[key] = value

    def add_round(self, key, value):
        self.results_dict.setdefault(key, [])
        self.results_dict[key].append(value)


