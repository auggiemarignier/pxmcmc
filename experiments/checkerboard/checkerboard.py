import numpy as np
import healpy as hp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from pylab import cm


def pixels_in_range(lat, lon, lat_range, lon_range, Nside=32):
    pixels = []
    for lt in np.arange(lat - lat_range / 2, lat + lat_range / 2):
        if lt < -90:
            lt = -90
        if lt > 90:
            lt = 90
        for ln in np.arange(lon - lon_range / 2, lon + lon_range / 2):
            if ln < -180:
                ln += 360
            if ln > 180:
                ln -= 360
            pixels.append(hp.ang2pix(Nside, ln, lt, lonlat=True))
    return pixels


def pixelise(signal, Nside, longs, lats):
    Npix = hp.nside2npix(Nside)
    pixnum = hp.ang2pix(Nside, longs, lats, lonlat=True)
    amap = np.zeros(Npix)
    count = np.zeros(Npix)
    nsample = len(signal)
    for i in range(nsample):
        pix = pixnum[i]
        amap[pix] += signal[i]
        count[pix] += 1.0
    for i in range(Npix):
        if count[i] > 0:
            amap[i] = amap[i] / count[i]
        else:
            amap[i] = hp.UNSEEN
    return amap


def build_bivariate_normal_pdf(x_range, y_range, mean=[0.0, 0.0], cov=np.eye(2)):
    x, y = np.meshgrid(np.linspace(-1, 1, x_range), np.linspace(-1, 1, y_range))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal(mean, cov)
    pdf = rv.pdf(pos)
    return pdf


class CheckerBoard:
    def __init__(self, base_size=15, Nside=32, gaussian=True, mean=None, cov=None):
        self.base_size = base_size
        self.gaussian = gaussian
        self.Nside = Nside
        if self.gaussian:
            assert mean is not None
            assert cov is not None
            self.mean = mean
            self.cov = cov

    def build_base_board(self):
        if self.gaussian:
            n_in_row = 360 // self.base_size
            n_in_col = 180 // self.base_size

            pdf = build_bivariate_normal_pdf(
                20 * self.base_size, 20 * self.base_size, self.mean, self.cov
            )

            row = np.hstack([pdf, -pdf] * (n_in_row // 2))
            base_board = np.vstack([row, -row] * (n_in_col // 2))

            lons = np.linspace(-180, 180, base_board.shape[1])
            lats = np.linspace(-90, 90, base_board.shape[0])
            lons, lats = np.meshgrid(lons, lats)
            lons, lats = lons.flatten(), lats.flatten()
            base_board = pixelise(base_board.flatten(), self.Nside, lons, lats)
        else:
            lats = np.arange(-90, 90 + self.base_size, self.base_size)
            lons = np.arange(-180, 180 + self.base_size, self.base_size)
            base_board = np.ones(hp.nside2npix(self.Nside))
            i = 0
            for lat in lats:
                for lon in lons:
                    i += 1
                    pixels = pixels_in_range(
                        lat, lon, self.base_size, self.base_size, Nside=self.Nside
                    )
                    if i % 2 == 0:
                        base_board[pixels] = -1
        self.board = base_board

    def add_feature(self, feature, lat, lon):
        """
        Feature is an array of shape (lat_step, lon_step)
        TODO: see if this works with different Nsides
        """
        lat_step, lon_step = feature.shape
        pixels = pixels_in_range(lat, lon, lat_step, lon_step, Nside=self.Nside)
        self.board[pixels] = feature


if __name__ == "__main__":
    gaussian_board = CheckerBoard(
        base_size=15, Nside=512, mean=[0.0, 0.0], cov=np.eye(2) * 0.15
    )
    gaussian_board.build_base_board()
    hp.write_map(
        f"./chkrbrd{gaussian_board.base_size}.fits",
        gaussian_board.board,
        overwrite=True,
    )

    hp.mollview(gaussian_board.board, cmap=cm.seismic_r, flip="geo")
    hp.graticule(gaussian_board.base_size)
    plt.savefig(f"./chkrbrd{gaussian_board.base_size}.png")
