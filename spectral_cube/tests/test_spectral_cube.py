import pytest
import operator
import itertools

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import numpy as np

from .. import SpectralCube, BooleanArrayMask, FunctionMask, LazyMask, CompositeMask

from . import path
from .helpers import assert_allclose, assert_array_equal



def cube_and_raw(filename):
    p = path(filename)

    d = fits.getdata(p)

    c = SpectralCube.read(p, format='fits')
    return c, d


class BaseTest(object):

    def setup_method(self, method):
        c, d = cube_and_raw('adv.fits')
        mask = BooleanArrayMask(d > 0.5, c._wcs)
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
        assert_allclose(c._get_filled_data(), expected)

    @pytest.mark.parametrize(('file', 'view'), (
                             ('adv.fits', np.s_[:, :,:]),
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
            assert_allclose(result, expected)

    @pytest.mark.parametrize('view', (np.s_[:, :,:],
                             np.s_[:2, :3, ::2]))
    def test_world_transposes_3d(self, view):
        c1, d1 = cube_and_raw('adv.fits')
        c2, d2 = cube_and_raw('vad.fits')

        for w1, w2 in zip(c1.world[view], c2.world[view]):
            assert_allclose(w1, w2)

    @pytest.mark.parametrize('view',
                             (np.s_[:, :,:],
                              np.s_[:2, :3, ::2],
                              np.s_[::3, ::2, :1],
                              np.s_[:], ))
    def test_world_transposes_4d(self, view):
        c1, d1 = cube_and_raw('advs.fits')
        c2, d2 = cube_and_raw('sadv.fits')
        for w1, w2 in zip(c1.world[view], c2.world[view]):
            assert_allclose(w1, w2)


    @pytest.mark.parametrize(('name','masktype'),
                             itertools.product(('advs', 'dvsa', 'sdav', 'sadv', 'vsad', 'vad', 'adv',),
                                               (BooleanArrayMask, LazyMask, FunctionMask, CompositeMask))
                            )
    def test_with_spectral_unit(self, name, masktype):
        cube, data = cube_and_raw(name + '.fits')
        cube_freq = cube.with_spectral_unit(u.Hz)

        if masktype == BooleanArrayMask:
            mask = BooleanArrayMask(data>0, wcs=cube._wcs)
        elif masktype == LazyMask:
            mask = LazyMask(lambda x: x>0, cube=cube)
        elif masktype == FunctionMask:
            mask = FunctionMask(lambda x: x>0)
        elif masktype == CompositeMask:
            mask1 = FunctionMask(lambda x: x>0)
            mask2 = LazyMask(lambda x: x>0, cube)
            mask = CompositeMask(mask1, mask2)

        cube2 = cube.with_mask(mask)
        cube_masked_freq = cube2.with_spectral_unit(u.Hz)

        assert cube_freq._wcs.wcs.ctype[cube_freq._wcs.wcs.spec] == 'FREQ-W2F'
        assert cube_masked_freq._wcs.wcs.ctype[cube_masked_freq._wcs.wcs.spec] == 'FREQ-W2F'
        assert cube_masked_freq._mask._wcs.wcs.ctype[cube_masked_freq._mask._wcs.wcs.spec] == 'FREQ-W2F'

        # values taken from header
        rest = 1.42040571841E+09*u.Hz
        crval = -3.21214698632E+05*u.m/u.s
        outcv = crval.to(u.m, u.doppler_optical(rest)).to(u.Hz, u.spectral())

        assert_allclose(cube_freq._wcs.wcs.crval[cube_freq._wcs.wcs.spec],
                        outcv.to(u.Hz).value)
        assert_allclose(cube_masked_freq._wcs.wcs.crval[cube_masked_freq._wcs.wcs.spec],
                        outcv.to(u.Hz).value)
        assert_allclose(cube_masked_freq._mask._wcs.wcs.crval[cube_masked_freq._mask._wcs.wcs.spec],
                        outcv.to(u.Hz).value)


class TestFilters(BaseTest):

    def test_mask_data(self):
        c, d = self.c, self.d
        expected = np.where(d > .5, d, np.nan)
        assert_allclose(c._get_filled_data(), expected)

        expected = np.where(d > .5, d, 0)
        assert_allclose(c._get_filled_data(fill=0), expected)

    def test_flatten(self):
        c, d = self.c, self.d
        expected = d[d > 0.5]
        assert_allclose(c.flattened(), expected)

    def test_flatten_weights(self):
        c, d = self.c, self.d
        expected = d[d > 0.5] ** 2
        assert_allclose(c.flattened(weights=d), expected)

    @pytest.mark.xfail
    def test_slice(self):
        c, d = self.c, self.d

        expected = d[:3, :2, ::2]
        expected = expected[expected > 0.5]
        assert_allclose(c[0:3, 0:2, 0::2].flattened(), expected)


class TestNumpyMethods(BaseTest):

    def _check_numpy(self, cubemethod, array, func):
        for axis in [None, 0, 1, 2]:
            for how in ['auto', 'slice', 'cube', 'ray']:
                expected = func(array, axis=axis)
                actual = cubemethod(axis=axis)
                assert_allclose(actual, expected)

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
        assert_allclose(self.c.median(axis=0), m)

    def test_percentile(self):
        m = np.empty(self.d.sum(axis=0).shape)
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                ray = self.d[:, y, x]
                ray = ray[ray > 0.5]
                m[y, x] = np.percentile(ray, 3)
        assert_allclose(self.c.percentile(3, axis=0), m)

    @pytest.mark.parametrize('method', ('sum', 'min', 'max',
                             'median', 'argmin', 'argmax'))
    def test_transpose(self, method):
        c1, d1 = cube_and_raw('adv.fits')
        c2, d2 = cube_and_raw('vad.fits')
        for axis in [None, 0, 1, 2]:
            assert_allclose(getattr(c1, method)(axis=axis),
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

        with pytest.raises(u.UnitsError) as exc:
            self.c.closest_spectral_channel(1 * u.s)
        assert exc.value.args[0] == "'value' should be in frequency equivalent or velocity units (got s)"

        with pytest.raises(u.UnitsError) as exc:
            self.c.closest_spectral_channel(1. * u.Hz)
        assert exc.value.args[0] == "Spectral axis is in velocity units and 'value' is in frequency-equivalent units - use SpectralCube.with_spectral_unit first to convert the cube to frequency-equivalent units, or search for a velocity instead"

    def test_slab(self):
        ms = u.m / u.s
        c2 = self.c.spectral_slab(-320000 * ms, -318600 * ms)
        assert_allclose(c2._data, self.d[1:3])
        assert c2._mask is not None

    def test_slab_reverse_limits(self):
        ms = u.m / u.s
        c2 = self.c.spectral_slab(-318600 * ms, -320000 * ms)
        assert_allclose(c2._data, self.d[1:3])
        assert c2._mask is not None

    def test_slab_preserves_wcs(self):
        # regression test
        ms = u.m / u.s
        crpix = list(self.c._wcs.wcs.crpix)
        self.c.spectral_slab(-318600 * ms, -320000 * ms)
        assert list(self.c._wcs.wcs.crpix) == crpix


class TestRepr(BaseTest):

    def test_repr(self):
        assert repr(self.c) == """
SpectralCube with shape=(4, 3, 2):
 n_x: 2  type_x: RA---SIN  unit_x: deg
 n_y: 3  type_y: DEC--SIN  unit_y: deg
 n_s: 4  type_s: VOPT      unit_s: m / s
        """.strip()

    def test_repr_withunit(self):
        self.c._unit = u.Jy
        assert repr(self.c) == """
SpectralCube with shape=(4, 3, 2) and unit=Jy:
 n_x: 2  type_x: RA---SIN  unit_x: deg
 n_y: 3  type_y: DEC--SIN  unit_y: deg
 n_s: 4  type_s: VOPT      unit_s: m / s
        """.strip()


def test_yt():
    cube = SpectralCube.read(path('adv.fits'))
    # Without any special arguments
    ds1 = cube.to_yt()
    # With spectral factor = 0.5
    spectral_factor = 0.5
    ds2 = cube.to_yt(spectral_factor=spectral_factor)
    # With nprocs = 4
    nprocs = 4
    ds3 = cube.to_yt(nprocs=nprocs)
    # The following assertions just make sure everything is
    # kosher with the datasets generated in different ways
    assert_array_equal(ds1.domain_dimensions, ds2.domain_dimensions)
    assert_array_equal(ds2.domain_dimensions, ds3.domain_dimensions)
    assert_allclose(ds1.domain_left_edge, ds2.domain_left_edge)
    assert_allclose(ds2.domain_left_edge, ds3.domain_left_edge)
    assert_allclose(ds1.domain_width, ds2.domain_width*np.array([1,1,1.0/spectral_factor]))
    assert_allclose(ds1.domain_width, ds3.domain_width)
    assert nprocs == len(ds3.index.grids)
    assert ds1.spec_cube
    assert ds2.spec_cube
    assert ds3.spec_cube
    # Now check that we can compute quantities of the flux
    # and that they are equal
    dd1 = ds1.all_data()
    dd2 = ds2.all_data()
    dd3 = ds3.all_data()
    flux1_tot = dd1.quantities.total_quantity("flux")
    flux2_tot = dd2.quantities.total_quantity("flux")
    flux3_tot = dd3.quantities.total_quantity("flux")
    flux1_min, flux1_max = dd1.quantities.extrema("flux")
    flux2_min, flux2_max = dd2.quantities.extrema("flux")
    flux3_min, flux3_max = dd3.quantities.extrema("flux")
    assert flux1_tot == flux2_tot
    assert flux1_tot == flux3_tot
    assert flux1_min == flux2_min
    assert flux1_min == flux3_min
    assert flux1_max == flux2_max
    assert flux1_max == flux3_max
    # Now test round-trip conversions between yt and world coordinates
    yt_coord1 = ds1.domain_left_edge + np.random.random(size=3)*ds1.domain_width
    world_coord1 = cube.yt2world(yt_coord1)
    assert_allclose(cube.world2yt(world_coord1), yt_coord1)
    yt_coord2 = ds2.domain_left_edge + np.random.random(size=3)*ds2.domain_width
    world_coord2 = cube.yt2world(yt_coord2, spectral_factor=0.5)
    assert_allclose(cube.world2yt(world_coord2, spectral_factor=0.5), yt_coord2)
    yt_coord3 = ds3.domain_left_edge + np.random.random(size=3)*ds3.domain_width
    world_coord3 = cube.yt2world(yt_coord3)
    assert_allclose(cube.world2yt(world_coord3), yt_coord3)

def test_read_write_rountrip(tmpdir):
    cube = SpectralCube.read(path('adv.fits'))
    tmp_file = str(tmpdir.join('test.fits'))
    cube.write(tmp_file)
    cube2 = SpectralCube.read(tmp_file)

    assert cube.shape == cube.shape
    assert_allclose(cube._data, cube2._data)
    assert cube._wcs.to_header_string() == cube2._wcs.to_header_string()


def _dummy_cube():
    data = np.array([[[0, 1, 2, 3, 4]]])
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO-HEL']

    def lower_threshold(data, wcs, view=()):
        return data[view] > 0

    m1 = FunctionMask(lower_threshold)

    cube = SpectralCube(data, wcs=wcs, mask=m1)
    return cube


def test_with_mask():

    def upper_threshold(data, wcs, view=()):
        return data[view] < 3

    m2 = FunctionMask(upper_threshold)

    cube = _dummy_cube()
    cube2 = cube.with_mask(m2)

    assert_allclose(cube._get_filled_data(), [[[np.nan, 1, 2, 3, 4]]])
    assert_allclose(cube2._get_filled_data(), [[[np.nan, 1, 2, np.nan, np.nan]]])


def test_with_mask_with_boolean_array():
    cube = _dummy_cube()
    mask = cube._data > 2
    cube2 = cube.with_mask(mask, inherit_mask=False)
    assert isinstance(cube2._mask, BooleanArrayMask)
    assert cube2._mask._wcs is cube._wcs
    assert cube2._mask._mask is mask


def test_with_mask_with_bad_array_shape():
    cube = _dummy_cube()
    mask = np.zeros((5, 5), dtype=np.bool)
    with pytest.raises(ValueError) as exc:
        cube.with_mask(mask)
    assert exc.value.args[0] == ("Mask shape doesn't match data shape: "
                                 "(5, 5) vs (1, 1, 5)")


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
        assert_allclose(actual, expected)


def test_preserve_spectral_unit():
    # astropy.wcs has a tendancy to change spectral units from e.g. km/s to
    # m/s, so we have a workaround - check that it works.

    cube, data = cube_and_raw('advs.fits')

    cube_freq = cube.with_spectral_unit(u.GHz)
    assert cube_freq.wcs.wcs.cunit[2] == 'Hz'  # check internal
    assert cube_freq.spectral_axis.unit is u.GHz

    # Check that this preferred unit is propagated
    new_cube = cube_freq.with_fill_value(fill_value=3.4)
    assert new_cube.spectral_axis.unit is u.GHz
