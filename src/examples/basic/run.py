from fado.arguments.arguments import AttackArguments
from fado.orchestrate import prepare_orchestrate
from fado.data.downloader import leaf_executor

config_path = 'config/attack_config.yaml'

args = AttackArguments(config_path)
leaf_executor(args.dataset)
#prepare_orchestrate(config_path, args, dev=True)
