import pytest
import operator

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import numpy as np

from spectral_cube import SpectralCube, SpectralCubeMask, FunctionMask, read
from . import path


def cube_and_raw(filename):
    p = path(filename)

    d = fits.getdata(p)

    c = read(p, format='fits')
    return c, d

def assert_almost_equal(arr1,arr2):
    if hasattr(arr1,'to') and hasattr(arr2,'to'):
        x = arr1.to(arr2.unit)
        np.testing.assert_array_almost_equal_nulp(x.value,arr2.value)
    else:
        np.testing.assert_array_almost_equal_nulp(arr1,arr2)

class BaseTest(object):

    def setup_method(self, method):
        c, d = cube_and_raw('adv.fits')
        mask = SpectralCubeMask(d > 0.5, c._wcs)
        c._mask = mask
        self.c = c
        self.mask = mask
        self.d = d


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
        np.testing.assert_allclose(c._get_filled_data(), expected)

    @pytest.mark.parametrize(('file', 'view'), (
                             ('adv.fits', np.s_[:, :, :]),
                             ('adv.fits', np.s_[::2, :, :2]),
                             ('adv.fits', np.s_[0]),
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
        world = [w[view] * u.Unit(wcs.wcs.cunit[i])
                 for i, w in enumerate(world)][::-1]

        w2 = c.world[view]
        for result, expected in zip(w2, world):
            np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize('view', (np.s_[:, :, :],
                             np.s_[:2, :3, ::2]))
    def test_world_transposes_3d(self, view):
        c1, d1 = cube_and_raw('adv.fits')
        c2, d2 = cube_and_raw('vad.fits')

        for w1, w2 in zip(c1.world[view], c2.world[view]):
            np.testing.assert_allclose(w1, w2)

    @pytest.mark.parametrize('view',
                             (np.s_[:, :, :],
                              np.s_[:2, :3, ::2],
                              np.s_[::3, ::2, :1],
                              np.s_[:], ))
    def test_world_transposes_4d(self, view):
        c1, d1 = cube_and_raw('advs.fits')
        c2, d2 = cube_and_raw('sadv.fits')
        for w1, w2 in zip(c1.world[view], c2.world[view]):
            np.testing.assert_allclose(w1, w2)


class TestFilters(BaseTest):

    def test_mask_data(self):
        c, d = self.c, self.d
        expected = np.where(d > .5, d, np.nan)
        np.testing.assert_allclose(c._get_filled_data(), expected)

        expected = np.where(d > .5, d, 0)
        np.testing.assert_allclose(c._get_filled_data(fill=0), expected)

    def test_flatten(self):
        c, d = self.c, self.d
        expected = d[d > 0.5]
        np.testing.assert_allclose(c.flattened(), expected)

    def test_flatten_weights(self):
        c, d = self.c, self.d
        expected = d[d > 0.5] ** 2
        np.testing.assert_allclose(c.flattened(weights=d), expected)

    @pytest.mark.xfail
    def test_slice(self):
        c, d = self.c, self.d

        expected = d[:3, :2, ::2]
        expected = expected[expected > 0.5]
        np.testing.assert_allclose(c[0:3, 0:2, 0::2].flattened(), expected)


class TestNumpyMethods(BaseTest):

    def _check_numpy(self, cubemethod, array, func):
        for axis in [None, 0, 1, 2]:
            expected = func(array, axis=axis)
            actual = cubemethod(axis=axis)
            np.testing.assert_allclose(actual, expected)

    def test_sum(self):
        d = np.where(self.d > 0.5, self.d, np.nan)
        self._check_numpy(self.c.sum, d, np.nansum)

    def test_max(self):
        d = np.where(self.d > 0.5, self.d, np.nan)
        self._check_numpy(self.c.max, d, np.nanmax)

    def test_min(self):
        d = np.where(self.d > 0.5, self.d, np.nan)
        self._check_numpy(self.c.min, d, np.nanmin)

    def test_argmax(self):
        d = np.where(self.d > 0.5, self.d, -10)
        self._check_numpy(self.c.argmax, d, np.nanargmax)

    def test_argmin(self):
        d = np.where(self.d > 0.5, self.d, 10)
        self._check_numpy(self.c.argmin, d, np.nanargmin)

    def test_median(self):
        m = np.empty(self.d.sum(axis=0).shape)
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                ray = self.d[:, y, x]
                ray = ray[ray > 0.5]
                m[y, x] = np.median(ray)
        np.testing.assert_allclose(self.c.median(axis=0), m)

    def test_percentile(self):
        m = np.empty(self.d.sum(axis=0).shape)
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                ray = self.d[:, y, x]
                ray = ray[ray > 0.5]
                m[y, x] = np.percentile(ray, 3)
        np.testing.assert_allclose(self.c.percentile(3, axis=0), m)

    @pytest.mark.parametrize('method', ('sum', 'min', 'max',
                             'median', 'argmin', 'argmax'))
    def test_transpose(self, method):
        c1, d1 = cube_and_raw('adv.fits')
        c2, d2 = cube_and_raw('vad.fits')
        for axis in [None, 0, 1, 2]:
            np.testing.assert_allclose(getattr(c1, method)(axis=axis),
                                       getattr(c2, method)(axis=axis))


class TestSlab(BaseTest):

    def test_closest_spectral_channel(self):
        c = self.c
        ms = u.m / u.s
        assert c.closest_spectral_channel(-321214.698632 * ms) == 0
        assert c.closest_spectral_channel(-319926.48366321 * ms) == 1
        assert c.closest_spectral_channel(-318638.26869442 * ms) == 2

        assert c.closest_spectral_channel(-320000 * ms) == 1
        assert c.closest_spectral_channel(-340000 * ms) == 0
        assert c.closest_spectral_channel(0 * ms) == 3

    def test_spectral_channel_bad_units(self):
        with pytest.raises(u.UnitsError):
            self.c.closest_spectral_channel(0 * u.s, rest_frequency=1 / u.s)

        with pytest.raises(u.UnitsError):
            self.c.closest_spectral_channel(0 * u.s)

    def test_slab(self):
        ms = u.m / u.s
        c2 = self.c.spectral_slab(-320000 * ms, -318600 * ms)
        np.testing.assert_allclose(c2._data, self.d[1:3])
        assert c2._mask is not None

    def test_slab_reverse_limits(self):
        ms = u.m / u.s
        c2 = self.c.spectral_slab(-318600 * ms, -320000 * ms)
        np.testing.assert_allclose(c2._data, self.d[1:3])
        assert c2._mask is not None

def test_read_write_rountrip():
    cube = read(path('adv.fits'))
    cube.write(path('test.fits'),clobber=True)
    cube2 = read(path('test.fits'))

    assert cube.shape == cube.shape
    np.testing.assert_allclose(cube._data, cube2._data)
    assert cube._wcs.to_header_string() == cube2._wcs.to_header_string()


def test_with_mask():

    data = np.array([[[0,1,2,3,4]]])

    def lower_threshold(data, wcs, view=()):
        return data[view] > 0

    def upper_threshold(data, wcs, view=()):
        return data[view] < 3

    m1 = FunctionMask(lower_threshold)
    m2 = FunctionMask(upper_threshold)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO-HEL']

    cube = SpectralCube(data, wcs=wcs, mask=m1)
    cube2 = cube.with_mask(m2)

    np.testing.assert_allclose(cube._get_filled_data(), [[[np.nan, 1, 2, 3, 4]]])
    np.testing.assert_allclose(cube2._get_filled_data(), [[[np.nan, 1, 2, np.nan, np.nan]]])

    def test_slab_preserves_wcs(self):
        # regression test
        ms = u.m / u.s
        crpix = list(self.c._wcs.wcs.crpix)
        self.c.spectral_slab(-318600 * ms, -320000 * ms)
        assert list(self.c._wcs.wcs.crpix) == crpix


class TestMasks(BaseTest):

    @pytest.mark.parametrize('op', (operator.gt, operator.lt,
                             operator.le, operator.ge))
    def test_operator_threshold(self, op):

        # choose thresh to exercise proper equality tests
        thresh = self.d.ravel()[0]
        m = op(self.c, thresh)
        self.c._mask = m

        expected = self.d[op(self.d, thresh)]
        actual = self.c.flattened()
        np.testing.assert_array_equal(actual, expected)
