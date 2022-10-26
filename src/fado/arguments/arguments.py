import random

import yaml


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

    def __contains__(self, key):
        return hasattr(self, key)
