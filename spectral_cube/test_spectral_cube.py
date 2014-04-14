import os

import pytest
from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

from spectral_cube import SpectralCube, SpectralCubeMask


def path(filename):
    return os.path.join('test_data', filename)


def cube_and_raw(filename):
    p = path(filename)

    d = fits.getdata(p)
    wcs = WCS(p)

    c = SpectralCube(d, wcs)
    assert c._data.ndim == 4
    return c, d


@pytest.mark.parametrize(('name', 'trans'), (
                         ('advs', [0, 1, 2, 3]),
                         ('dvsa', [2, 3, 0, 1]),
                         ('sdav', [0, 2, 1, 3]),
                         ('sadv', [0, 1, 2, 3]),
                         ('vsad', [3, 0, 1, 2]),
                         ('vad', [2, 0, 1]),
                         ('adv', [0, 1, 2]),
                         ))
def test_consistent_transposition(name, trans):
    """data() should return velocity axis first, then world 1, then world 0"""

    c, d = cube_and_raw(name + '.fits')
    expected = np.squeeze(d.transpose(trans))
    np.testing.assert_array_equal(c.get_data(), expected)


def test_mask_data():
    p = path('advs.fits')
    d = fits.getdata(p)
    wcs = WCS(p)
    mask = SpectralCubeMask(d > .1, wcs)
    c = SpectralCube(d, wcs, mask)

    expected = np.where(d > .1, d, np.nan)
    np.testing.assert_array_equal(c.get_data(), expected)

    expected = np.where(d > .1, d, 0)
    np.testing.assert_array_equal(c.get_data(fill=0), expected)
