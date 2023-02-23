# Initialize logger
import logging
import os

logger = logging.getLogger("fado")
rank = os.getenv('FADO_ID')
rank_str = ""
if rank:
    rank_str = f" [Rank:{rank}]"
format_str = logging.Formatter(fmt=f"[%(asctime)s] [%(levelname)s] [%(node_id)s] "
                                   "[%(filename)s:%(lineno)d:%(funcName)s] %("
                                   "message)s",
                               datefmt="%Y-%m-%d %H:%M:%S")
stdout_handle = logging.StreamHandler()
stdout_handle.setFormatter(format_str)
logger.setLevel(logging.INFO)
logger.addHandler(stdout_handle)
