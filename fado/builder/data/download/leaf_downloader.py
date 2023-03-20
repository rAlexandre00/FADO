import logging
import os
from pathlib import Path

import requests
import py7zr

from fado.builder.data.download.downloader import Downloader
from fado.cli.arguments.arguments import FADOArguments
from fado.constants import ALL_DATA_FOLDER, TEMP_DIRECTORY

FEMNIST_URL = 'https://cloud.lasige.di.fc.ul.pt/index.php/s/XyyjGxmZaMgbeJD/download/femnist.7z'

fado_args = FADOArguments()
DATA_FOLDER = os.path.join(ALL_DATA_FOLDER, fado_args.dataset)

download_module_path = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger('fado')
logger = logging.LoggerAdapter(logger, {'node_id': 'builder'})


class LEAFDownloader(Downloader):

    def download(self):
        if fado_args.dataset == 'femnist':
            download_emnist()
        else:
            raise Exception("femnist dataset not supported yet")


def download_emnist():
    assert fado_args.dataset == 'femnist'
    download_dataset(FEMNIST_URL)


def download_dataset(url):
    if Path(DATA_FOLDER).is_dir():
        logger.info('femnist dataset already downloaded')
        return
    logger.info('Downloading femnist dataset')
    response = requests.get(url, verify=False)
    os.makedirs(TEMP_DIRECTORY, exist_ok=True)
    archive_path = os.path.join(TEMP_DIRECTORY, "data.7z")
    open(archive_path, "wb").write(response.content)
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=DATA_FOLDER)
    os.remove(os.path.join(TEMP_DIRECTORY, "data.7z"))
