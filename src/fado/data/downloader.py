import os
import shutil
import subprocess

from fado import leaf

DATASETS = ['femnist', 'shakespeare', 'sent140', 'celeba']

def leaf_executor(args):

    """
    The LEAF's module inside FADO contains, for each different dataset,
    the scripts that download and process the data.

    This method copies those scripts into a temporary directory that is created
    in the current working directory called 'temp_leaf'. It is not a good practice
    to download and process data directly in the module's directory.

    After executing the script that downloads and generates the data (preprocess), 
    the data that is generated is copied to the data directory used by FADO (./data/all).

    Data from different datasets is organized inside ./data/all under different directories
    to provide agility in experimenting with different datasets without having to download
    and process them over and over again.

    TODO ability to change dataset size (--sf parameter)
    """

    dataset = args.dataset

    if dataset not in DATASETS:
        raise Exception(f"Dataset {dataset} not supported! Choose one of the following: {DATASETS}")

    temp_dir = os.path.abspath(os.path.join('.', 'temp_leaf'))
    data_fado = os.path.abspath(os.path.join('.', args.all_data_folder, dataset))

    if not os.path.exists(data_fado):
        os.makedirs(data_fado, exist_ok=True)
    else:
        print(f'Dataset for {dataset} already generated. Proceeding.')
        return

    # Getting LEAF's module path
    leaf_path = os.path.dirname(leaf.__file__)

    # Copying LEAF's utils folder to temp directory
    shutil.copytree(os.path.join(leaf_path, 'utils'), os.path.join(temp_dir, 'utils'))
    # Copying LEAF's dataset folder to temp directory
    shutil.copytree(os.path.join(leaf_path, dataset), os.path.join(temp_dir, dataset), dirs_exist_ok=True)

    curr_dir = os.getcwd() # will be useful to return to current directory
    os.chdir(os.path.join(temp_dir, dataset)) # changing current directory

    # Calling LEAF script
    subprocess.call(['./preprocess.sh', f'-s niid', f'--sf 0.05', f'-k 0', f'-t sample'])

    # Remove unnecessary folders
    data_path = os.path.join(temp_dir, dataset, 'data')
    subfolders = [ f.path for f in os.scandir(data_path) if f.is_dir() and f.name not in ['train', 'test']]
    for subfolder in subfolders:
        shutil.rmtree(subfolder, ignore_errors=True)

    # Copying train and test folder to the .{all_data_folder}/{dataset} folder
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    shutil.copytree(train_dir, os.path.join(data_fado, 'train'))
    shutil.copytree(test_dir, os.path.join(data_fado, 'test'))

    # Clean temporary directory and reset directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)   

    os.chdir(curr_dir) 
    