FADO_VERSION=$(cat fado/constants/__init__.py | grep FADO_VERSION | sed -n -e 's/^\(.*\)\(FADO_VERSION = "\)\(.*\)"$/\3/p')

echo "Updating fado on pypi"
python setup.py sdist
twine upload dist/FAaDO-$FADO_VERSION*

echo "Updating fado images on Docker Hub"

cd docker/requirements/
./upload_image.sh $FADO_VERSION
cd ../node/
./upload_image.sh $FADO_VERSION
cd ../router/
./upload_image.sh $FADO_VERSION

