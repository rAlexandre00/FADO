#!/usr/bin/env bash

server="172.20.1.0"

# Sleep while server is starting
while true
do
ping -c1 -W1 -q $server &>/dev/null && break || sleep 1
done

# Sleep while server has not ended the training
while true
do
ping -c1 -W1 -q $server &>/dev/null && sleep 1 || break
done