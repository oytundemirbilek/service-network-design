# AirNet

Welcome to AirNet ! A service network optimization model for evTOL air taxis for urban transportation in New York City.

## Installation

You can install the module and its dependencies as a Python package simply via pip:

```sh
pip install -e .
```

The module is also designed to be OS-independent and supports python>=3.9. Note that you need to have a valid gurobi license to use to module.

## Usage

AirNet includes a command line interface (CLI) which provides parameterization of the main utilities. You can print the help message explaining all the parameters:

```sh
airnet --optimize service
```

A sensitivity analysis will run the optimization model for multiple parameters and saves the results as a csv file:

```sh
airnet --sensitivity all
```

There is also an hub location model provided; simply selects a number of hubs which optimizes the distances to the other vertiports:

```sh
airnet --optimize hubs
```


## File explanation

1. data_analysis.ipynb: A notebook to provide analytical functions for the optimization results.

2. service_network.ipynb: A notebook to provide a rather more interactive interface for running the model.

## Authors

Ahkyoung Kim

Oytun Demirbilek
