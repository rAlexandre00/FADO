from typing import Type

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.fl.aggregate.base_aggregator import Aggregator
from fado.runner.fl.aggregate.mean_aggregator import MeanAggregator


class AggregatorManager:

    @classmethod
    def get_aggregator(cls, global_module) -> Aggregator:
        args = FADOArguments()
        if args.aggregator == 'mean':
            return MeanAggregator(global_module)
        elif '.py' in args.aggregator:
            return args.get_class('aggregator')(global_module)
        else:
            raise Exception("Specified aggregator does not exist")
        pass
