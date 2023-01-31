# FADO - Federated Attack and Defense Orchestrator

## User interaction


### Build

These commands build the prerequisites for running FADO

- fado build data (Download LEAF dataset)
- fado build partitions (Split the data)
- fado build compose (Create docker-compose.yaml)
---
- 'fado build' performs the three actions above
---

Options for build:

- -d <dataset>
- -nb <number of benign clients>
- -nm <number of malign clients>

### Deploy

- fado run

### Clean

- fado clean