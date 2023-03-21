from fado.cli.arguments.arguments import FADOArguments
from fado.security.attack.network.nlafl_attacker import NLAFLAttacker

fado_args = FADOArguments('/app/config/fado_config.yaml')


class NetworkAttackerManager:

    def __init__(self):
        pass

    @classmethod
    def get_attacker(cls, model, attacker_test_x, attacker_test_y):

        network_attack_name = fado_args.network_attack if 'network_attack' in fado_args else None

        if network_attack_name == 'nlafl':
            return NLAFLAttacker(model, attacker_test_x, attacker_test_y)
        else:
            return None

