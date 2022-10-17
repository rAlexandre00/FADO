#!/usr/bin/env bash

python3 torch_client.py --cf config/fedml_config_malicious.yaml --rank $FEDML_RANK --role client
