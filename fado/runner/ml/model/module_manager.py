from fado.cli.arguments.arguments import FADOArguments


class ModelManager:

    @classmethod
    def get_model(cls):
        args = FADOArguments()
        if args.model in ['nlafl_emnist_torch', 'nlafl_emnist_tf', 'nlafl_fashionmnist_tf', 'nlafl_dbpedia_tf']:
            if args.model == 'nlafl_emnist_torch':
                from fado.runner.ml.model.built_in.nlafl_emnist_torch import NlaflEmnistTorch
                return NlaflEmnistTorch()
            elif args.model == 'nlafl_emnist_tf':
                from fado.runner.ml.model.built_in.nlafl_emnist_tf import NlaflEmnistTf
                return NlaflEmnistTf()
            elif args.model == 'nlafl_fashionmnist_tf':
                from fado.runner.ml.model.built_in.nlafl_fashionmnist_tf import NlaflFashionmnistTf
                return NlaflFashionmnistTf()
            elif args.model == 'nlafl_dbpedia_tf':
                from fado.runner.ml.model.built_in.nlafl_dbpedia_tf import NlaflDbpediaTf
                return NlaflDbpediaTf()
        elif args.model in ['mnist_conv_torch']:
            from fado.runner.ml.model.built_in.mnist_conv_torch import MnistConvTorch
            return MnistConvTorch()
        elif '.py' in args.model:
            return args.get_class('model')()
        else:
            raise Exception(f"Model {args.model} not found")
