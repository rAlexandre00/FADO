import logging
import os
from pathlib import Path

import requests
import py7zr

from fado.builder.data.download.downloader import Downloader
from fado.cli.arguments.arguments import FADOArguments
from fado.constants import ALL_DATA_FOLDER, TEMP_DIRECTORY

EMNIST_URL = 'https://cloud.lasige.di.fc.ul.pt/index.php/s/t49CoempxyzAC2F/download/emnist.7z'
FASHION_MNIST_URL = 'https://cloud.lasige.di.fc.ul.pt/index.php/s/bEezR8DoQH4rJiq/download/fashionMnist.7z'
DBPEDIA_URL = 'https://cloud.lasige.di.fc.ul.pt/index.php/s/kmcCz8N7836jFd6/download/dbpedia.7z'

fado_args = FADOArguments()
DATA_FOLDER = os.path.join(ALL_DATA_FOLDER, fado_args.dataset)

download_module_path = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'builder'})


class NLAFLDownloader(Downloader):

    def download(self):
        if fado_args.dataset == 'nlafl_emnist':
            download_emnist()
        else:
            raise Exception("NLAFL dataset not supported yet")


def download_emnist():
    assert fado_args.dataset == 'nlafl_emnist'
    download_dataset(EMNIST_URL)


def download_dataset(url):
    if Path(DATA_FOLDER).is_dir():
        logger.info('nlafl emnist dataset already downloaded')
        return
    logger.info('Downloading nlafl emnist dataset')
    response = requests.get(url, verify=False)
    os.makedirs(TEMP_DIRECTORY, exist_ok=True)
    archive_path = os.path.join(TEMP_DIRECTORY, "data.7z")
    open(archive_path, "wb").write(response.content)
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=DATA_FOLDER)
    os.remove(os.path.join(TEMP_DIRECTORY, "data.7z"))
