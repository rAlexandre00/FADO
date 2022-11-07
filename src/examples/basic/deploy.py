import os
from fado.constants import FADO_DIR
import subprocess

os.chdir(FADO_DIR)
subprocess.run(['docker', 'compose', 'build'])
subprocess.run(['docker', 'compose', 'up'])