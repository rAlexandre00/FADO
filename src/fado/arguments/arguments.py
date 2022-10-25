import random

import yaml
from fado.security.utils import load_attack_class, load_defense_class


class AttackArguments:
    """ A class for reading arguments from a yaml file """

    def __init__(self, config_path):
        """
            Parameters:
                config_path(str): path of yaml configuration file
        """
        with open(config_path, 'r') as file:
            args = yaml.load(file, Loader=yaml.FullLoader)

        self._set_arguments(args)
        self._process_arguments()

    def _process_arguments(self):
        """ Uses the arguments read to do needed computations"""
        if 'random_seed' in self:
            random.seed(self.random_seed)

    def _set_arguments(self, key_pairs):
        """ Sets the arguments

            Parameters:
                key_pairs(dict): key, value pairs with (property_name, property_value)
        """
        for key, value in key_pairs.items():
            setattr(self, key, value)

        # If the argument 'attack_spec' is specified, load its contents to the main arguments scope
        if hasattr(self, "attack_spec"):
            if '.yaml' in self.attack_spec:
                with open(self.attack_spec, 'r') as file:
                    configuration = yaml.safe_load(file)

                for arg_key, arg_val in configuration.items():
                    setattr(self, arg_key, arg_val)
            else:
                setattr(self, "attack_spec", load_attack_class(self))

        # If the argument 'defense_spec' is specified, load its contents to the main arguments scope
        if hasattr(self, "defense_spec"):
            if '.yaml' in self.defense_spec:
                with open(self.defense_spec, 'r') as file:
                    configuration = yaml.safe_load(file)

                for arg_key, arg_val in configuration.items():
                    setattr(self, arg_key, arg_val)
            else:
                setattr(self, "defense_spec", load_defense_class(self))

    def __contains__(self, key):
        return hasattr(self, key)
