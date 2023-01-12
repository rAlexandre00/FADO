#!/usr/bin/env bash

readarray -d , -t benign_ranks < config/benign_ranks.csv
readarray -d , -t malicious_ranks < config/malicious_ranks.csv

for rank in "${benign_ranks[@]}"
do
  env FEDML_RANK="$rank" /app/run_client.sh &
done

for rank in "${malicious_ranks[@]}"
do
  env FEDML_RANK="$rank" /app/run_client.sh &
done

wait
