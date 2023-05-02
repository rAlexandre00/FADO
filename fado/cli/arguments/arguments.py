import importlib
import logging
import os
import random
import sys
from pathlib import Path

import yaml

from fado.constants import IMPORT_OUT

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'arguments'})

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

            config_path = Path(config_path).parent.absolute().__str__()
            setattr(self, 'config_path', config_path)
            self._set_arguments(args, config_path)
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
        if 'python_import_folder' in self:
            sys.path.append(os.path.join(IMPORT_OUT))
            sys.path.append('/app/import')

    def _set_arguments(self, key_pairs, config_path):
        """ Sets the arguments
            Parameters:
                key_pairs(dict): key, value pairs with sections(dicts) that contain n (property_name, property_value)
        """
        for section_name, section in key_pairs.items():
            if type(section) == dict:
                self._set_arguments(section, config_path)
            elif type(section) == list:
                if 'vary' not in self:
                    self.vary = {}
                self.vary[section_name] = section
            else:
                setattr(self, section_name, section)

    def set_argument(self, key, value):
        setattr(self, key, value)

    def get_argument(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def save_to_file(self, file_path):
        with open(file_path, 'w') as file:
            yaml.dump(self.__dict__, file)

    def get_class(self, key):
        module_name = os.path.join(self.get_argument(key))[:-3]
        module = importlib.import_module(module_name)
        return module.get_class()
