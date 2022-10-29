#!/usr/bin/env bash

# Sleep while server is starting
while true
do
ping -c1 -W1 -q $server &>/dev/null && break || sleep 1
done

/bin/bash apply_nat.sh

# Sleep while server has not ended the training
while true
do
ping -c1 -W1 -q $server &>/dev/null && sleep 1 || break
done
