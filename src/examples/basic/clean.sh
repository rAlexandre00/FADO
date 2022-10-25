#!/bin/bash

docker stop $(docker ps -a -q) 2> /dev/null
docker rm $(docker ps -a -q) 2> /dev/null
#docker rmi $(docker images -q)
rm -rf certs
sudo find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | sudo xargs rm -rf
rm -rf docker
rm -rf data/partitions
sudo rm -rf runs
rm -rf logs
rm .config_hash 2> /dev/null
rm docker-compose.yml 2> /dev/null
rm config/fedml* 2> /dev/null
rm config/grpc* 2> /dev/null
echo 'Clean finished!'
