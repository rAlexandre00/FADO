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
        self.malicious_clients = key_pairs['malicious_clients']
        self.benign_clients = key_pairs['benign_clients']
        self.grpc_ipconfig_out = key_pairs['grpc_ipconfig_out']
        self.fedml_config_out = key_pairs['fedml_config_out']
        self.fedml_config_out_malicious = key_pairs['fedml_config_out_malicious']
        self.docker_compose_out = key_pairs['docker_compose_out']
        self.all_data_folder = key_pairs['all_data_folder']
        self.partition_data_folder = key_pairs['partition_data_folder']
        self.model_file = key_pairs['model_file']
        self.attack_spec = key_pairs['attack_spec']
        self.defense_spec = key_pairs['defense_spec']
        self.random_seed = key_pairs['random_seed']

    def __contains__(self, key):
        return hasattr(self, key)
