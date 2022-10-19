
from setuptools import setup, find_packages

requirements = [
    "grpcio",
    "fedml",
    "PyYAML",
    "pyOpenSSL"
]

setup(
    name='FAaDO',
    version='0.0.2',
    license='MIT',
    maintainer="Rodrigo Simoes",
    maintainer_email='rsimoes@lasige.di.fc.ul.pt',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/rAlexandre00/FADO',
    keywords='Federated Attack and Defense Orchestrator',
    install_requires=requirements

)
