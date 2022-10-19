#!/usr/bin/env bash

server="172.20.1.0"

# Sleep while server is starting
while true
do
if nc -z $server 8890 ; then
    break
else
    sleep 1
fi
done

# Sleep while server has not ended the training
while true
do
if nc -z $server 8890 ; then
    sleep 1
else
    break
fi
done