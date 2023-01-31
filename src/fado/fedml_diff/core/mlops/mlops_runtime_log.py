import json
import logging
import os
from logging import handlers


# class MLOpsRuntimeLog:

def init_logs(self, show_stdout_log=True):
    log_file_path, program_prefix = build_log_file_path(self.args)
    logging.raiseExceptions = True
    self.logger = logging.getLogger(log_file_path)
    format_str = logging.Formatter(fmt="[" + program_prefix + "] [%(asctime)s] [%(levelname)s] "
                                                              "[%(filename)s:%(lineno)d:%(funcName)s] %("
                                                              "message)s",
                                   datefmt="%a, %d %b %Y %H:%M:%S")
    stdout_handle = logging.StreamHandler()
    stdout_handle.setFormatter(format_str)
    if show_stdout_log:
        stdout_handle.setLevel(logging.INFO)
        self.logger.setLevel(logging.INFO)
    else:
        stdout_handle.setLevel(logging.CRITICAL)
        self.logger.setLevel(logging.CRITICAL)
    self.logger.addHandler(stdout_handle)
    if hasattr(self, "should_write_log_file") and self.should_write_log_file:
        when = 'D'
        backup_count = 100
        file_handle = handlers.TimedRotatingFileHandler(filename=log_file_path, when=when,
                                                        backupCount=backup_count, encoding='utf-8')
        file_handle.setFormatter(format_str)
        file_handle.setLevel(logging.INFO)
        self.logger.addHandler(file_handle)
    logging.root = self.logger


def build_log_file_path(in_args):
    if in_args.rank == 0:
        if hasattr(in_args, "server_id"):
            edge_id = in_args.server_id
        else:
            if hasattr(in_args, "edge_id"):
                edge_id = in_args.edge_id
            else:
                edge_id = 0
        program_prefix = "FedML-Server({}) @device-id-{}".format(in_args.rank, edge_id)
    else:
        if hasattr(in_args, "client_id"):
            edge_id = in_args.client_id
        elif hasattr(in_args, "client_id_list"):
            edge_ids = json.loads(in_args.client_id_list)
            if len(edge_ids) > 0:
                edge_id = edge_ids[0]
            else:
                edge_id = 0
        else:
            if hasattr(in_args, "edge_id"):
                edge_id = in_args.edge_id
            else:
                edge_id = 0
        program_prefix = "FedML-Client({rank}) @device-id-{edge}".format(
            rank=in_args.rank, edge=edge_id
        )

    os.system("mkdir -p " + in_args.log_file_dir)
    log_file_path = os.path.join(in_args.log_file_dir, "fedml-run-"
                                 + str(in_args.run_id)
                                 + "-edge-"
                                 + str(edge_id)
                                 + ".log")

    return log_file_path, program_prefix
