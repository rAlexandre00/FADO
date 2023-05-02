cp requirements-incomplete.txt requirements.txt
echo FAaDO==$1 >> requirements.txt

docker build -t ralexandre00/fado-node-requirements:$1 .
docker push ralexandre00/fado-node-requirements:$1

rm requirements.txt
