
from setuptools import setup, find_packages
# from distutils.extension import Extension
# from Cython.Build import cythonize


requirements = [
    "grpcio",
    "fedml",
    "PyYAML",
    "pyOpenSSL",
    "flask",
    "Cython"
]

# ext_modules = Extension('fado.fedml_diff.core.distributed.communication.grpc.binder',
#                         sources=['src/fado/fedml_diff/core/distributed/communication/grpc/binder.pyx'],
#                         extra_compile_args=['-std=c++11'],
#                         language='c++',
#                         )

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
