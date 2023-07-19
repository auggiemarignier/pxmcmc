# Earth Topography

A simple example that effectively just finds the wavelet coefficients of a map of Earth's topography using proximal MCMC.

https://github.com/auggiemarignier/pxmcmc/assets/42379892/5992aad6-23eb-47cc-acf8-cf19a0ea3572

Has the option of adding noise to the original image to make the problem more realistic.
Can also create a block of very noisy data covering the African continent, resulting in higher uncertainty in the MCMC results in that area.

## Quickstart

If you just want to get something, run the following from this directory with your environment activated

```bash
python main.py --infile ETOPO1_Ice_hpx_256.fits
python plot.py myula_synethesis_<timestamp>.hdf5 .
```

This will take a minute or so and will not converge, although you will start to see very rough outlines of the continents in the output plots.
To run for longer, modify the parameters in `main.py` line 123.
For the plotting command, replace the input filename accordingly.

## Command Line Arguments

```text
usage: main.py [-h] [--infile INFILE] [--outdir OUTDIR] [--jobid JOBID] [--algo ALGO] [--setting SETTING] [--delta DELTA] [--mu MU] [--L L] [--makenoise] [--sigma SIGMA]
               [--scaleafrica SCALEAFRICA]

options:
  -h, --help            show this help message and exit
  --infile INFILE       Path to input datafile.
  --outdir OUTDIR       Output directory. Default '.'.
  --jobid JOBID         Optional ID that will be added to the end of the output filename. Default '0'.
  --algo ALGO           PxMCMC algorithm to be used. One of ['myula', 'pxmala', 'skrock']. Default 'myula'.
  --setting SETTING     'synthesis' or 'analysis'. Default 'myula'.
  --delta DELTA         PxMCMC step size. Default 1e-6
  --mu MU               Regularisation parameter (prior width). Default 1.
  --L L                 Angular bandlimit. Default 32.
  --makenoise           Add noise to data.
  --sigma SIGMA         Noise level to be added to data.
  --scaleafrica SCALEAFRICA
                        Factor by which to increase the noise level in Africa.
```

```text
usage: plot.py [-h] [--suffix SUFFIX] [--burn BURN] [--save_npy] datafile directory

positional arguments:
  datafile         Path to .hdf5 file with pxmcmc results.
  directory        Directory in which to save plots.

options:
  -h, --help       show this help message and exit
  --suffix SUFFIX  Optional suffix to output filenames.
  --burn BURN      Ignore the first <burn> MCMC samples. Default 0.
  --save_npy       Also save the output summary maps as .npy files
```
