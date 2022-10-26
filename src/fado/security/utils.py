import yaml
import pkgutil
import fado.security.attack.factory as attack_factory
import fado.security.defense.factory as defense_factory
from fado.security.constants import ATTACK_BYZANTINE, DEFENSE_KRUM
from fado.security.attack.factory import *
from fado.security.defense.factory import *


def load_attack(args):
    # If the argument 'attack_spec' is specified, load its contents to the main arguments scope
    if hasattr(args, "attack_spec"):
        if '.yaml' in args.attack_spec:
            with open(args.attack_spec, 'r') as file:
                configuration = yaml.safe_load(file)

            for arg_key, arg_val in configuration.items():
                setattr(args, arg_key, arg_val)
        else:
            setattr(args, "attack_spec", _load_attack_class(args))


def load_defense(args):
    """ This method should only be called at runtime
        It reads the configuration file and loads a class dynamically
    """
    # If the argument 'defense_spec' is specified, load its contents to the main arguments scope
    if hasattr(args, "defense_spec"):
        if '.yaml' in args.defense_spec:
            with open(args.defense_spec, 'r') as file:
                configuration = yaml.safe_load(file)

            for arg_key, arg_val in configuration.items():
                setattr(args, arg_key, arg_val)
        else:
            setattr(args, "defense_spec", _load_defense_class(args))


def _load_attack_class(args):
    """ This method should only be called at runtime
        It reads the configuration file and loads a class dynamically
    """
    if args.attack_spec == ATTACK_BYZANTINE:
        stream = pkgutil.get_data(attack_factory.__name__, 'byzantine_attack_config.yaml')
        configuration = _load_yaml_config_stream(stream)
        for arg_key, arg_val in configuration.items():
            setattr(args, arg_key, arg_val)
        return ByzantineAttack
    else:
        raise Exception("Attack class not found")


def _load_defense_class(args):
    """ This method should only be called at runtime
        It reads the configuration file and loads a class dynamically
    """
    if args.defense_spec == DEFENSE_KRUM:
        stream = pkgutil.get_data(defense_factory.__name__, 'krum_config.yaml')
        configuration = _load_yaml_config_stream(stream)
        for arg_key, arg_val in configuration.items():
            setattr(args, arg_key, arg_val)
        return KrumDefense
    else:
        raise Exception("Defense class not found")


def _load_yaml_config_stream(stream):
    try:
        return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError("Yaml error - check yaml file")
