from fado.cli.arguments.arguments import FADOArguments
from fado.runner.ml.model.built_in.nlafl_emnist import NlaflEmnist


class ModelManager:

    @classmethod
    def get_model(cls):
        args = FADOArguments()
        if args.model in ['nlafl_emnist']:
            if args.model == 'nlafl_emnist':
                return NlaflEmnist()
        else:
            raise Exception(f"Model {args.model} not found")
