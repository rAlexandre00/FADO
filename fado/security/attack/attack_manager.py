from typing import Type

from fado.cli.arguments.arguments import FADOArguments
from fado.runner.fl.aggregate.base_aggregator import Aggregator
from fado.runner.fl.aggregate.mean_aggregator import MeanAggregator
from fado.security.attack.attack_base import Attack
from fado.security.attack.model.random_attacker import RandomAttacker


class AttackManager:

    @classmethod
    def get_attacker(cls) -> Attack:
        args = FADOArguments()
        if args.attacker_args.name == 'random':
            return RandomAttacker()
        else:
            return Attack()
        pass
