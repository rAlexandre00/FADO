#!/usr/bin/env bash

python3 torch_client.py --cf config/fedml_config.yaml --rank $FEDML_RANK --role client
