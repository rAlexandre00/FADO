docker build -t ralexandre00/fado-node:$1 --build-arg VERSION=$1 .
docker push ralexandre00/fado-node:$1
