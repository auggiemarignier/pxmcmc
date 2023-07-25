# Gravitational Weak Lensing

Solves the mass-mapping problem from weak lensing data.

The input data file is a few GB, so run

``` bash
python download_takahasi.py
```

first to obtain the data.

```text
usage: main.py [-h] [--outdir OUTDIR] [--jobid JOBID] [--algo ALGO] [--setting SETTING] [--delta DELTA] [--mu MU] [--L L] infile

positional arguments:
  infile             A fits file containing the kappa ground truth in healpix format

options:
  -h, --help         show this help message and exit
  --outdir OUTDIR    Output directory. Default '.'.
  --jobid JOBID      Optional ID that will be added to the end of the output filename. Default '0'.
  --algo ALGO        PxMCMC algorithm to be used. One of ['myula', 'pxmala', 'skrock']. Default 'myula'.
  --setting SETTING  'synthesis' or 'analysis'. Default 'myula'.
  --delta DELTA      PxMCMC step size. Default 1e-6
  --mu MU            Regularisation parameter (prior width). Default 1.
  --L L              Angular bandlimit. Default 512.
```

```text
usage: plot.py [-h] [--suffix SUFFIX] [--burn BURN] [--save_npy] [--no-mask] datafile directory

positional arguments:
  datafile         Path to .hdf5 file with pxmcmc results.
  directory        Directory in which to save plots.

options:
  -h, --help       show this help message and exit
  --suffix SUFFIX  Optional suffix to output filenames.
  --burn BURN      Ignore the first <burn> MCMC samples. Default 100.
  --save_npy       Also save the output summary maps as .npy files
  --no-mask        Reveal what is behind the Euclid mask
```
