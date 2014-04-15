import os

import pytest
from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

from spectral_cube import SpectralCube, SpectralCubeMask, read


def path(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)


def cube_and_raw(filename):
    p = path(filename)

    d = fits.getdata(p)

    c = read(p, format='fits')
    return c, d


class TestSpectralCube(object):

    @pytest.mark.parametrize(('name', 'trans'), (
                             ('advs', [0, 1, 2, 3]),
                             ('dvsa', [2, 3, 0, 1]),
                             ('sdav', [0, 2, 1, 3]),
                             ('sadv', [0, 1, 2, 3]),
                             ('vsad', [3, 0, 1, 2]),
                             ('vad', [2, 0, 1]),
                             ('adv', [0, 1, 2]),
                             ))
    def test_consistent_transposition(self, name, trans):
        """data() should return velocity axis first, then world 1, then world 0"""
        c, d = cube_and_raw(name + '.fits')
        expected = np.squeeze(d.transpose(trans))
        np.testing.assert_array_equal(c.get_data(), expected)

    def test_mask_data(self):
        c, d = cube_and_raw('advs.fits')
        d = d[0]
        wcs = c._wcs
        mask = SpectralCubeMask(d > .1, wcs)
        c._mask = mask

        expected = np.where(d > .1, d, np.nan)
        np.testing.assert_array_equal(c.get_data(), expected)

        expected = np.where(d > .1, d, 0)
        np.testing.assert_array_equal(c.get_data(fill=0), expected)

    @pytest.mark.parametrize(('file', 'view'), (
                             ('adv.fits', np.s_[:, :, :]),
                             ('adv.fits', np.s_[::2, :, :2]),
                             ))
    def test_world(self, file, view):
        p = path(file)
        d = fits.getdata(p)
        wcs = WCS(p)
        c = SpectralCube(d, wcs)

        shp = d.shape
        inds = np.indices(d.shape)
        pix = np.column_stack([i.ravel() for i in inds[::-1]])
        world = wcs.all_pix2world(pix, 0).T

        world = [w.reshape(shp) for w in world]
        world = [w[view] for w in world][::-1]

        w2 = c.world[view]
        for result, expected in zip(w2, world):
            np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize('view', (np.s_[:, :, :],
                             np.s_[:2, :3, ::2]))
    def test_world_transposes(self, view):
        c1, d1 = cube_and_raw('adv.fits')
        c2, d2 = cube_and_raw('vad.fits')

        for w1, w2 in zip(c1.world[view], c2.world[view]):
            np.testing.assert_array_equal(w1, w2)

"""
TODO:

 check that pix<->world is correct for all transpositions

 """
