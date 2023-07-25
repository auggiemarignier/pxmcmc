[![PyPI version](https://badge.fury.io/py/pxmcmc.svg)](https://badge.fury.io/py/pxmcmc) [![Documentation Status](https://readthedocs.org/projects/pxmcmc/badge/?version=latest)](https://pxmcmc.readthedocs.io/en/latest/?badge=latest) [![test](https://github.com/auggiemarignier/pxmcmc/actions/workflows/python-app.yml/badge.svg)](https://github.com/auggiemarignier/pxmcmc/actions/workflows/python-app.yml) [![status](https://joss.theoj.org/papers/ed274b8490fbc89365e6e0a993f73d86/status.svg)](https://joss.theoj.org/papers/ed274b8490fbc89365e6e0a993f73d86)

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

Examples of how to use this code with sample data are found in the `experiments` directory.
Please start with the `earthtopography` example, which will quickly run something to get you going!

```bash
cd experiments/earthtopography
python main.py --infile ETOPO1_Ice_hpx_256.fits
python plot.py myula_synthesis_<timestamp>.hdf5 .
```

The `phasevel` and `weaklensing` examples replicate the work shown [in this paper](https://doi.org/10.1093/rasti/rzac010).

## Contributing

Contributions to the package are encouraged! If you wish to contribute, are experiencing problems with the code or need further support, please open an [issue](https://github.com/auggiemarignier/pxmcmc/issues/new) to start a discussion.  Changes will be integrated via pull requests.

## CITATION
If you use this package in your work please cite the following papers

Marignier (2023) PxMCMC: A Python package for proximal Markov Chain Monte Carlo, Journal of Open Source Software, 0(0), 5582. https://doi.org/10.xxxxxx

Marignier et al., Posterior sampling for inverse imaging problems on the sphere in seismology and cosmology, RAS Techniques and Instruments, Volume 2, Issue 1, January 2023, Pages 20–32, https://doi.org/10.1093/rasti/rzac010
