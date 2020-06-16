import pytest
import numpy as np

from pxmcmc.utils import GreatCirclePath

# Tests based on example given on Great Circle Navigation Wikipedia page


@pytest.fixture
def path(Nside):
    start = (-33, -71.6)
    stop = (31.4, 121.8)
    return GreatCirclePath(start, stop, Nside)


def test_gcp_course(path):
    assert np.round(np.rad2deg(path._course_at_start()), 2) == -94.41
    assert np.round(np.rad2deg(path._course_at_end()), 2) == -78.42
    assert np.round(np.rad2deg(path._course_at_node()), 2) == -56.74


def test_gcp_epicentral_distance(path):
    assert np.round(np.rad2deg(path._epicentral_distance()), 2) == 168.56
    assert np.round(np.rad2deg(path._node_to_start()), 2) == -96.76


def test_gcp_midpoint(path):
    colat, lon = np.rad2deg(path._point_at_fraction(0.5))
    lat = 90 - colat
    assert np.round(lat, 2) == -6.81
    assert np.round(lon, 2) == -159.18
