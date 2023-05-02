import logging
from typing import Type

from fado.cli.arguments.arguments import FADOArguments
from fado.security.attack.client.attack_base import Attack
from fado.security.attack.client.random_attacker import RandomAttacker
from fado.security.attack.client.nlafl_poison_attacker import NLAFLPoisonAttacker

logger = logging.getLogger("fado")
logger = logging.LoggerAdapter(logger, {'node_id': 'none'})


class ClientAttackManager:

    @classmethod
    def get_attacker(cls, client_id) -> Attack:
        args = FADOArguments()

        model_attack_name = args.model_attack_name if 'model_attack_name' in args else None

        if model_attack_name == 'random':
            return RandomAttacker()
        elif model_attack_name == 'nlafl_poison':
            return NLAFLPoisonAttacker(client_id=client_id)
        elif 'py' in model_attack_name:
            return args.get_class('model_attack_name')()
        else:
            raise Exception(f"Model {model_attack_name} not found")
