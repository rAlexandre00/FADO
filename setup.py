
from setuptools import setup, find_packages


requirements = [
    "torch==1.13.1",
    "tensorflow==2.11.0",
    "PyYAML==6.0",
    "py7zr==0.20.4",
    "numpy==1.21",
    "pandas==1.5.2"
]

setup(
    name='FAaDO',
    version='0.0.3',
    license='MIT',
    maintainer="Rodrigo Simoes",
    maintainer_email='rsimoes@lasige.di.fc.ul.pt',
    packages=find_packages('.'),
    package_dir={'': '.'},
    url='https://github.com/rAlexandre00/FADO',
    keywords='Federated Attack and Defense Orchestrator',
    install_requires=requirements,
    package_data={
        "": ["*.yaml"]
    },
    entry_points={
        'console_scripts': [
            'fado = fado.cli.fado_run:cli',
        ],
    },
    # ext_modules=cythonize(ext_modules, language_level = "3"),
)
