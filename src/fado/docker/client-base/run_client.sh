#!/usr/bin/env bash

CUDA_LAUNCH_BLOCKING=1 CUDA_MODULE_LOADING=LAZY python3 torch_client.py --cf "config/user_$FEDML_RANK/fedml_config.yaml" --rank "$FEDML_RANK" --role client