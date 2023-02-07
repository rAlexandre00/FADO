#!/usr/bin/env bash

readarray -d , -t benign_ranks < config/benign_ranks.csv
readarray -d , -t malicious_ranks < config/malicious_ranks.csv

num_ben="${#benign_ranks[@]}"
num_mal="${#malicious_ranks[@]}"

total_ranks=$(($num_ben + $num_mal))

ips=()
for device_ip in {1..254}.{0..255}
do
  ips[${#ips[@]}]="${device_ip}"
  if [[ ${#ips[@]} -eq $total_ranks ]]; then
    break
  fi
done

for rank in "${benign_ranks[@]}"
do
  env FEDML_RANK="$rank" ip netns exec ns10.1."${ips[rank-1]}" /app/run_client.sh &
done

for rank in "${malicious_ranks[@]}"
do
  env FEDML_RANK="$rank" ip netns exec ns10.1."${ips[rank-1]}" /app/run_client.sh &
done

wait

