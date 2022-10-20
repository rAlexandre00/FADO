import yaml


class AttackArguments:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            args = yaml.load(file, Loader=yaml.FullLoader)
        self.malicious_clients = args['malicious_clients']
        self.benign_clients = args['benign_clients']
        self.grpc_ipconfig_out = args['grpc_ipconfig_out']
        self.fedml_config_out = args['fedml_config_out']
        self.fedml_config_out_malicious = args['fedml_config_out_malicious']
        self.docker_compose_out = args['docker_compose_out']
        self.all_data_folder = args['all_data_folder']
        self.partition_data_folder = args['partition_data_folder']
        self.model_file = args['model_file']
        self.attack_spec = args['attack_spec']
        self.defense_spec = args['defense_spec']
