# Seismic Phase Velocity

Inverts path-average phase velocity to obtain a global 2D map of phase velocity.

This is synthetic example, using [GDM52 at 40s](https://onlinelibrary.wiley.com/doi/10.1111/j.1365-246X.2011.05225.x) as the ground truth.
The ground truth has been bandlimited to $L=28$ and is found in `GDM40_L28.npy`.
The path average data, generated for a realistic set of seismic paths is found in `synthetic_GDM40_0S254_L28.txt` - this is the input data for the inversion.

```text
usage: main.py [-h] [--outdir OUTDIR] [--jobid JOBID] [--algo ALGO] [--setting SETTING] [--delta DELTA] [--mu MU] [--L L] [--eta ETA] [--nsim] infile pathsfile

positional arguments:
  infile             Path to input datafile.
  pathsfile          path to .npz file with scipy sparse matrix. If file is not found, sparse matrix will be generated and saved here.

options:
  -h, --help         show this help message and exit
  --outdir OUTDIR    Output directory. Default '.'.
  --jobid JOBID      Optional ID that will be added to the end of the output filename. Default '0'.
  --algo ALGO        PxMCMC algorithm to be used. One of ['myula', 'pxmala', 'skrock']. Default 'myula'.
  --setting SETTING  'synthesis' or 'analysis'. Default 'myula'.
  --delta DELTA      PxMCMC step size. Default 1e-6
  --mu MU            Regularisation parameter (prior width). Default 1.
  --L L              Angular bandlimit. Default 28.
  --eta ETA          Wavelet power decay factor. See pxmcmc.prior.S2_Wavelets_L1_Power_Weights. Default 1.
  --nsim             Applies wieghting for number of similar paths
```

```text
usage: plot.py [-h] [--suffix SUFFIX] [--burn BURN] [--save_npy] datafile directory

positional arguments:
  datafile         Path to .hdf5 file with pxmcmc results.
  directory        Directory in which to save plots.

options:
  -h, --help       show this help message and exit
  --suffix SUFFIX  Optional suffix to output filenames.
  --burn BURN      Ignore the first <burn> MCMC samples. Default 100.
  --save_npy       Also save the output summary maps as .npy files
```
