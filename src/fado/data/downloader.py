import os
import shutil

from fado import leaf

def leaf_executor(dataset):

    temp_dir = os.path.join('.','temp_leaf')
    data_fado = os.path.join('.', 'data', 'all')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    leaf_path = os.path.dirname(leaf.__file__)

    shutil.copytree(os.path.join(leaf_path, 'utils'), os.path.join(temp_dir, 'utils'))
    shutil.copytree(os.path.join(leaf_path, dataset), temp_dir, dirs_exist_ok=True)

    train_dir = os.path.join(temp_dir, 'data', 'train')
    test_dir = os.path.join(temp_dir, 'data', 'test')

    shutil.copytree(train_dir, data_fado)
    shutil.copytree(test_dir, data_fado)
    