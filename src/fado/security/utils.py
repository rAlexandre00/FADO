import yaml
import pkgutil
import fado.security.attack.factory as attack_factory
import fado.security.defense.factory as defense_factory
from fado.security.constants import ATTACK_BYZANTINE, DEFENSE_KRUM
from fado.security.attack.factory import *
from fado.security.defense.factory import *



def load_attack_class(args):

    if args.attack_spec == ATTACK_BYZANTINE:
        stream = pkgutil.get_data(attack_factory.__name__, 'byzantine_attack_config.yaml')
        configuration = load_yaml_config_stream(stream)
        for arg_key, arg_val in configuration.items():
            setattr(args, arg_key, arg_val)
        return ByzantineAttack
    else:
        raise Exception("Attack class not found")

def load_defense_class(args):

    if args.defense_spec == DEFENSE_KRUM:
        stream = pkgutil.get_data(defense_factory.__name__, 'krum_config.yaml')
        configuration = load_yaml_config_stream(stream)
        for arg_key, arg_val in configuration.items():
            setattr(args, arg_key, arg_val)
        return KrumDefense
    else:
        raise Exception("Defense class not found")


def load_yaml_config_stream(stream):
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError("Yaml error - check yaml file")
