#!/usr/bin/env bash

CUDA_LAUNCH_BLOCKING=1 python3 torch_client.py --cf config/fedml_config_malicious.yaml --rank $FEDML_RANK --role client
