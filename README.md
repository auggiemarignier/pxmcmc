[![Documentation Status](https://readthedocs.org/projects/pxmcmc/badge/?version=latest)](https://pxmcmc.readthedocs.io/en/latest/?badge=latest) [![test](https://github.com/auggiemarignier/pxmcmc/actions/workflows/python-app.yml/badge.svg)](https://github.com/auggiemarignier/pxmcmc/actions/workflows/python-app.yml) [![status](https://joss.theoj.org/papers/ed274b8490fbc89365e6e0a993f73d86/status.svg)](https://joss.theoj.org/papers/ed274b8490fbc89365e6e0a993f73d86)

# Python ProxMCMC

## Installation

Available on [pypi](https://pypi.org/project/pxmcmc/)

```bash
pip install pxmcmc
```

If installing from source it recommended to use [poetry](https://python-poetry.org/)

```bash
git clone https://github.com/auggiemarignier/pxmcmc
cd pxmcmc
poetry install
source <ENVIRONMENT_LOCATION>/bin/activate
pytest
```

## Documentation

Full documentation available on [readthedocs](https://pxmcmc.readthedocs.io/en/latest/?badge=latest).

## Examples

Examples of how to use this code are found in the `experiments` directory.  Note that we don't provide any example input data files for the various experiments here, though can be made available upon request.

```bash
cd experiments/phasevelocity
python main.py --help
```
