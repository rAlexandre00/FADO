docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
#docker rmi $(docker images -q)
rm -rf certs
rm -rf __pycache__
rm -rf docker
rm -rf data/partitions
rm -rf runs
rm .config_hash
rm docker-compose.yml 2> /dev/null
rm config/fedml* 2> /dev/null
rm config/grpc* 2> /dev/null
echo 'Clean finished!'
