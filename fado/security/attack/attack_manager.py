from typing import Type

from fado.cli.arguments.arguments import FADOArguments
from fado.security.attack.attack_base import Attack
from fado.security.attack.model.random_attacker import RandomAttacker
from fado.security.attack.model.nlafl_poison_attacker import NLAFLPoisonAttacker


class AttackManager:

    @classmethod
    def get_attacker(cls, client_id) -> Attack:
        args = FADOArguments()

        model_attack_name = args.model_attack_name if 'model_attack_name' in args else None
        
        if model_attack_name == 'random':
            return RandomAttacker()
        elif model_attack_name == 'nlafl_poison':
            return NLAFLPoisonAttacker(client_id=client_id)
        else:
            return Attack()
