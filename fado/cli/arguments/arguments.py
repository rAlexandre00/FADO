import random

import yaml


class FADOArguments:
    """ A class for reading arguments from a yaml file """

    def __init__(self, config_path=None):
        """
            Parameters:
                config_path(str): path of yaml configuration file
        """
        if config_path is not None:
            with open(config_path, 'r') as file:
                args = yaml.load(file, Loader=yaml.FullLoader)

            self._set_arguments(args)
            self._process_arguments()

    def __new__(cls, config_path=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FADOArguments, cls).__new__(cls)
        return cls.instance

    def _process_arguments(self):
        """ Uses the arguments read to do needed computations"""
        if 'random_seed' in self:
            random.seed(self.random_seed)
            # TODO: check if TF or Torch is in use and set seed

    def _set_arguments(self, key_pairs):
        """ Sets the arguments
            Parameters:
                key_pairs(dict): key, value pairs with sections(dicts) that contain n (property_name, property_value)
        """
        for section_name, section in key_pairs.items():
            if type(section) == dict:
                self._set_arguments(section)
            else:
                setattr(self, section_name, section)

    def set_argument(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)
