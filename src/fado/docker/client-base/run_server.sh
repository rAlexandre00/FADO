#!/usr/bin/env bash

FEDML_RANK=0 CUDA_LAUNCH_BLOCKING=1 CUDA_MODULE_LOADING=LAZY python3 torch_server.py --cf "config/user_0/fedml_config.yaml" --rank 0 --role server
