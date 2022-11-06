#!/usr/bin/env bash

echo "Applying NAT"
/bin/bash apply_nat.sh

# Sleep while server is starting
while true
do
ping -c1 -W1 -q fado_server &>/dev/null && break || sleep 1
done

python3 fado_router.py &

# Sleep while server has not ended the training
while true
do
ping -c1 -W1 -q fado_server &>/dev/null && sleep 1 || break
done
