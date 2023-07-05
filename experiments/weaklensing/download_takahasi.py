"""
Downloads weak lensing maps from the Takahasi N-body simualtion
http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/

The conversion from binary to fits format is taken from
http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/read.py
"""
import requests
import os
import numpy as np
import healpy as hp
from tqdm import tqdm
import shutil


def binary_to_fits(filename):
    """
    Conversion from binary data to fits
    Taken from http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/read.py
    """
    skip = [0, 536870908, 1073741818, 1610612728, 2147483638, 2684354547, 3221225457]
    load_blocks = [skip[i + 1] - skip[i] for i in range(0, 6)]

    with open(filename, "rb") as f:
        rec = np.fromfile(f, dtype="uint32", count=1)[0]
        nside = np.fromfile(f, dtype="int32", count=1)[0]
        npix = np.fromfile(f, dtype="int64", count=1)[0]
        rec = np.fromfile(f, dtype="uint32", count=1)[0]
        print("nside:{} npix:{}".format(nside, npix))

        rec = np.fromfile(f, dtype="uint32", count=1)[0]

        kappa = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype="float32", count=blocks)
            np.fromfile(f, dtype="uint32", count=2)
            kappa = np.append(kappa, load)
            r = r - blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks) - 1:
                load = np.fromfile(f, dtype="float32", count=r)
                np.fromfile(f, dtype="uint32", count=2)
                kappa = np.append(kappa, load)

        gamma1 = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype="float32", count=blocks)
            np.fromfile(f, dtype="uint32", count=2)
            gamma1 = np.append(gamma1, load)
            r = r - blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks) - 1:
                load = np.fromfile(f, dtype="float32", count=r)
                np.fromfile(f, dtype="uint32", count=2)
                gamma1 = np.append(gamma1, load)

        gamma2 = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype="float32", count=blocks)
            np.fromfile(f, dtype="uint32", count=2)
            gamma2 = np.append(gamma2, load)
            r = r - blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks) - 1:
                load = np.fromfile(f, dtype="float32", count=r)
                np.fromfile(f, dtype="uint32", count=2)
                gamma2 = np.append(gamma2, load)

        omega = np.array([])
        r = npix
        for i, l in enumerate(load_blocks):
            blocks = min(l, r)
            load = np.fromfile(f, dtype="float32", count=blocks)
            np.fromfile(f, dtype="uint32", count=2)
            omega = np.append(omega, load)
            r = r - blocks
            if r == 0:
                break
            elif r > 0 and i == len(load_blocks) - 1:
                load = np.fromfile(f, dtype="float32", count=r)
                np.fromfile(f, dtype="uint32", count=2)
                omega = np.append(omega, load)

    print("loading completed")

    # example of saving data as a fits file
    hp.fitsfunc.write_map("output.fits", kappa)


def download_binary(filename):
    """
    Download original binary weak lensing data from
    http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/
    """
    ind = filename.find("nres")
    nres = int(filename[ind + 4 : ind + 6])
    url = f"http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub1/nres{nres}/{filename}"
    with requests.get(url, stream=True) as r:
        total_length = int(r.headers.get("Content-Length"))
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
            with open(filename, "wb") as file:
                shutil.copyfileobj(raw, file)


if __name__ == "__main__":
    nres = 12  # only options are 12, 13 or 14
    assert nres in [12, 13, 14]

    realisation = 0  # max is 107 (0 if nres=14)
    if nres in [12, 13]:
        assert (realisation >= 0) and (realisation <= 107)
    else:
        assert nres == 0

    redshift_id = 16  # between 1 and 66
    assert (redshift_id >= 1) and (redshift_id <= 66)

    filename_bin = f"allskymap_nres{nres}r{realisation:03}.zs{redshift_id}.mag.dat"
    filename_fits = (
        f"takahasi_{int(2**nres)}_{realisation:03}_zs{redshift_id}_kappa.fits"
    )
    if not os.path.exists(filename_fits):
        if not os.path.exists(filename_bin):
            print("Downloading Takahasi N-body simulation")
            download_binary(filename_bin)
        print(f"Converting binary file {filename_bin} to fits {filename_fits}")
        binary_to_fits(filename_bin)
        os.rename("output.fits", filename_fits)
    else:
        print(f"{filename_fits} already found!")
