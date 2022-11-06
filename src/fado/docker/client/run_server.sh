#!/usr/bin/env bash

CUDA_LAUNCH_BLOCKING=1 CUDA_MODULE_LOADING=LAZY python3 torch_server.py --cf config/fedml_config.yaml --rank 0 --role server
