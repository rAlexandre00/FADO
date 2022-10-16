import os

import wget
import zipfile

FEDML_DATA_MNIST_URL = "https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip"


def download_mnist(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    file_path = os.path.join(data_cache_dir, "MNIST.zip")

    # Download the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    if not os.path.exists(os.path.join(data_cache_dir, "all")):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_cache_dir)
        os.rename(os.path.join(data_cache_dir, "MNIST"), os.path.join(data_cache_dir, "all"))
