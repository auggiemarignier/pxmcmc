import numpy as np
import healpy as hp
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from pylab import cm
from astropy.coordinates import spherical_to_cartesian


def pixels_in_range(lat, lon, lat_range, lon_range, Nside=32, step=1):
    min_lat, max_lat = (lat - lat_range / 2, lat + lat_range / 2)
    min_lon, max_lon = (lon - lon_range / 2, lon + lon_range / 2)
    lats = np.deg2rad((min_lat, max_lat, max_lat, min_lat))
    lons = np.deg2rad((max_lon, max_lon, min_lon, min_lon))
    poly_xyz = np.array(spherical_to_cartesian(1, lats, lons)).T
    pixels = hp.query_polygon(Nside, poly_xyz)

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

    def add_feature(self, value, lat, lon, lat_range, lon_range):
        """
        Feature is centred at (lat, lon) with size (lat_range, lon_range).
        All the pixels in this range take the value value
        TODO: More complex features
        """
        pixels = pixels_in_range(lat, lon, lat_range, lon_range, Nside=self.Nside)
        self.board[pixels] = value


if __name__ == "__main__":
    gaussian_board = CheckerBoard(
        base_size=30, Nside=512, mean=[0.0, 0.0], cov=np.eye(2) * 0.15
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
