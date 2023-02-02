#!/usr/bin/env bash

# Sleep while server is starting
while true
do
echo 'trying to reach the server...'
ping -c1 -W1 -q fado_server &>/dev/null && break || sleep 1
done
echo 'reached the server...'

python3 fado_router.py &

# Sleep while server has not ended the training
while true
do
ping -c1 -W1 -q fado_server &>/dev/null && sleep 1 || break
done
