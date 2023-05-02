# FADO - Federated Attack and Defense Orchestrator

## Overview

Evaluations of current FL security mechanisms are often based 
on simplistic testing environments and  demand complex programming
to integrate new attacks/defenses.
FADO is an accessible platform that leverages a realistic environment to
facilitate the experimentation and evaluation of new solutions in
relevant FL scenarios. Comparison  with already proposed approaches
is also expedited since FADO provides a few out-of-the-box implementations.

## Dependencies

- Docker >= 20.0
- Python packages in setup.py

## Installation

### Using pip

```
pip install faado
```

### Using GitHub

``` 
git clone https://github.com/rAlexandre00/FADO.git
cd FADO
pip install -e .
```

## Usage

### Basic interaction

- ```fado``` - Performs all actions described in "Prepare", "Build", and "Run".

#### Flags needed
- ```-f <fado_config>``` - YAML file describing the emulation behaviour

### Prepare

Pulls required docker images to run FADO. Only needs to be run once.

### Build

These commands build the prerequisites for running a FADO emulation.
This step needs to be executed everytime a new configuration file is used.
If your fado configuration file contains varying parameters (yaml lists) you do not
have to execute this step since it will run automatically for every experiment.

- ```fado build``` - Performs the two actions below

- ```fado build download``` - Run Downloader (download dataset)
- ```fado build shape``` - Run Shaper (Make transformations to the data)

#### Flags needed
- ```-f <fado_config>``` - YAML file describing the emulation behaviour

### Run

- ```fado run``` - Runs the simulation according to the provided configuration file
- 
#### Flags needed
- ```-f <fado_config>``` - YAML file describing the emulation behaviour

### Clean
Commands that remove all files that FADO created/downloaded

- ```fado clean``` - Performs the two actions below
- ```fado clean prepare``` - Remove the pulled docker images
- ```fado clean build``` - Remove all files generated/downloaded while building simulations

## Examples

Some examples are provided at examples.

## Emulation Results

### Logs
``` ~/.fado/logs ```

### Results

Raw result files are located at: ``` ~/.fado/results ```

To print a table containing a mean of the results run: ```fado table```


