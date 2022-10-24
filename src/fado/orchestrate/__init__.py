import logging

from .orchestrator import prepare_orchestrate

# Initialize logger
logger = logging.getLogger("fado")
format_str = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] "
                                   "[%(filename)s:%(lineno)d:%(funcName)s] %("
                                   "message)s",
                               datefmt="%a, %d %b %Y %H:%M:%S")
stdout_handle = logging.StreamHandler()
stdout_handle.setFormatter(format_str)
logger.setLevel(logging.INFO)
logger.addHandler(stdout_handle)
