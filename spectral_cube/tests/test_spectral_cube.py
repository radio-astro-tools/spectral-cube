import pytest
import operator
import itertools
import warnings

# needed to test for warnings later
warnings.simplefilter('always', UserWarning)


from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs import _wcs
import numpy as np

from .. import (SpectralCube, BooleanArrayMask, FunctionMask, LazyMask,
                CompositeMask)
from ..spectral_cube import OneDSpectrum, Projection
from ..np_compat import allbadtonan
from .. import spectral_axis

from . import path
from .helpers import assert_allclose, assert_array_equal

from distutils.version import StrictVersion

try:
    import yt
    ytOK = True
    yt_version = (StrictVersion(yt.__version__[:-4])
                  if 'dev' in yt.__version__ else
                  StrictVersion(yt.__version__))
except ImportError:
    yt_version = StrictVersion('0.0.0')
    ytOK = False

try:
    import bottleneck
    bottleneckOK = True
except ImportError:
    bottleneckOK = False

try:
    import radio_beam
    RADIO_BEAM_INSTALLED = True
except ImportError:
    RADIO_BEAM_INSTALLED = False


def cube_and_raw(filename):
    p = path(filename)

    d = fits.getdata(p)

    c = SpectralCube.read(p, format='fits')
    return c, d


def test_arithmetic_warning(recwarn):

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    assert not cube._is_huge

    # make sure the small cube raises a warning about loading into memory
    cube + 5*cube.unit
    w = recwarn.list[-1]

    assert 'requires loading the entire cube into' in str(w.message)


def test_huge_disallowed():

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    data = np.empty([1e2,1e3,1e3])
    cube = SpectralCube(data=data, wcs=cube.wcs)

    assert cube._is_huge

    with pytest.raises(ValueError) as exc:
        cube + 5*cube.unit
    assert 'entire cube into memory' in exc.value.args[0]

    cube.allow_huge_operations = True
    # just make sure it doesn't fail
    cube + 5*cube.unit

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
                             ('vda', [0, 2, 1]),
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


    @pytest.mark.parametrize(('operation', 'value'),
                             ((operator.add, 0.5*u.K),
                              (operator.sub, 0.5*u.K),
                              (operator.mul, 0.5*u.K),
                              (operator.truediv, 0.5*u.K),
                              (operator.div if hasattr(operator,'div') else operator.floordiv, 0.5*u.K),
                             ))
    def test_apply_everywhere(self, operation, value):
        c1, d1 = cube_and_raw('advs.fits')

        # append 'o' to indicate that it has been operated on
        c1o = c1._apply_everywhere(operation, value)
        d1o = operation(u.Quantity(d1, u.K), value)

        assert np.all(d1o == c1o.filled_data[:])
        # allclose fails on identical data?
        #assert_allclose(d1o, c1o.filled_data[:])

    @pytest.mark.parametrize(('name', 'trans'), (
                             ('advs', [0, 1, 2, 3]),
                             ('dvsa', [2, 3, 0, 1]),
                             ('sdav', [0, 2, 1, 3]),
                             ('sadv', [0, 1, 2, 3]),
                             ('vsad', [3, 0, 1, 2]),
                             ('vad', [2, 0, 1]),
                             ('vda', [0, 2, 1]),
                             ('adv', [0, 1, 2]),
                             ))
    def test_getitem(self, name, trans):
        c, d = cube_and_raw(name + '.fits')

        expected = np.squeeze(d.transpose(trans))

        assert_allclose(c[0,:,:].value, expected[0,:,:])
        assert_allclose(c[:,:,0].value, expected[:,:,0])
        assert_allclose(c[:,0,:].value, expected[:,0,:])

        # Not implemented:
        #assert_allclose(c[0,0,:].value, expected[0,0,:])
        #assert_allclose(c[0,:,0].value, expected[0,:,0])
        assert_allclose(c[:,0,0].value, expected[:,0,0])

        assert_allclose(c[1,:,:].value, expected[1,:,:])
        assert_allclose(c[:,:,1].value, expected[:,:,1])
        assert_allclose(c[:,1,:].value, expected[:,1,:])

        # Not implemented:
        #assert_allclose(c[1,1,:].value, expected[1,1,:])
        #assert_allclose(c[1,:,1].value, expected[1,:,1])
        assert_allclose(c[:,1,1].value, expected[:,1,1])

        c2 = c.with_spectral_unit(u.km/u.s, velocity_convention='radio')

        assert_allclose(c2[0,:,:].value, expected[0,:,:])
        assert_allclose(c2[:,:,0].value, expected[:,:,0])
        assert_allclose(c2[:,0,:].value, expected[:,0,:])

        # Not implemented:
        #assert_allclose(c2[0,0,:].value, expected[0,0,:])
        #assert_allclose(c2[0,:,0].value, expected[0,:,0])
        assert_allclose(c2[:,0,0].value, expected[:,0,0])

        assert_allclose(c2[1,:,:].value, expected[1,:,:])
        assert_allclose(c2[:,:,1].value, expected[:,:,1])
        assert_allclose(c2[:,1,:].value, expected[:,1,:])

        # Not implemented:
        #assert_allclose(c2[1,1,:].value, expected[1,1,:])
        #assert_allclose(c2[1,:,1].value, expected[1,:,1])
        assert_allclose(c2[:,1,1].value, expected[:,1,1])

class TestArithmetic(object):

    def setup_method(self, method):
        self.c1, self.d1 = cube_and_raw('adv.fits')

        # make nice easy-to-test numbers
        self.d1.flat[:] = np.arange(self.d1.size)
        self.c1._data.flat[:] = np.arange(self.d1.size)

    @pytest.mark.parametrize(('value'),(1,1.0,2,2.0))
    def test_add(self,value):
        d2 = self.d1 + value
        c2 = self.c1 + value*u.K
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K

    def test_add_cubes(self):
        d2 = self.d1 + self.d1
        c2 = self.c1 + self.c1
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K

    @pytest.mark.parametrize(('value'),(1,1.0,2,2.0))
    def test_subtract(self, value):
        d2 = self.d1 - value
        c2 = self.c1 - value*u.K
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K

    def test_subtract_cubes(self):
        d2 = self.d1 - self.d1
        c2 = self.c1 - self.c1
        assert np.all(d2 == c2.filled_data[:].value)
        assert np.all(c2.filled_data[:].value == 0)
        assert c2.unit == u.K

    @pytest.mark.parametrize(('value'),(1,1.0,2,2.0))
    def test_mul(self, value):
        d2 = self.d1 * value
        c2 = self.c1 * value
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K

    def test_mul_cubes(self):
        d2 = self.d1 * self.d1
        c2 = self.c1 * self.c1
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K**2

    @pytest.mark.parametrize(('value'),(1,1.0,2,2.0))
    def test_div(self, value):
        d2 = self.d1 / value
        c2 = self.c1 / value
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K

    def test_div_cubes(self):
        d2 = self.d1 / self.d1
        c2 = self.c1 / self.c1
        assert np.all((d2 == c2.filled_data[:].value) | (np.isnan(c2.filled_data[:])))
        assert np.all((c2.filled_data[:] == 1) | (np.isnan(c2.filled_data[:])))
        assert c2.unit == u.dimensionless_unscaled

    @pytest.mark.parametrize(('value'),
                             (1,1.0,2,2.0))
    def test_pow(self, value):
        d2 = self.d1 ** value
        c2 = self.c1 ** value
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K**value

    def test_cube_add(self):
        c2 = self.c1 + self.c1
        d2 = self.d1 + self.d1
        assert np.all(d2 == c2.filled_data[:].value)
        assert c2.unit == u.K



class TestFilters(BaseTest):

    def test_mask_data(self):
        c, d = self.c, self.d
        expected = np.where(d > .5, d, np.nan)
        assert_allclose(c._get_filled_data(), expected)

        expected = np.where(d > .5, d, 0)
        assert_allclose(c._get_filled_data(fill=0), expected)

    @pytest.mark.parametrize('operation', (operator.lt, operator.gt, operator.le, operator.ge))
    def test_mask_comparison(self, operation):
        c, d = self.c, self.d
        dmask = operation(d, 0.6) & self.c.mask.include()
        cmask = operation(c, 0.6*u.K)
        assert (self.c.mask.include() & cmask.include()).sum() == dmask.sum()
        np.testing.assert_almost_equal(c.with_mask(cmask).sum().value,
                                       d[dmask].sum())

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
        self._check_numpy(self.c.sum, d, allbadtonan(np.nansum))
        # Need a secondary check to make sure it works with no
        # axis keyword being passed (regression test for issue introduced in
        # 150)
        assert np.all(self.c.sum().value == np.nansum(d))

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
SpectralCube with shape=(4, 3, 2) and unit=K:
 n_x:      2  type_x: RA---SIN  unit_x: deg    range:    24.062698 deg:   24.063349 deg
 n_y:      3  type_y: DEC--SIN  unit_y: deg    range:    29.934094 deg:   29.935209 deg
 n_s:      4  type_s: VOPT      unit_s: m / s  range:  -321214.699 m / s: -317350.054 m / s
        """.strip()

    def test_repr_withunit(self):
        self.c._unit = u.Jy
        assert repr(self.c) == """
SpectralCube with shape=(4, 3, 2) and unit=Jy:
 n_x:      2  type_x: RA---SIN  unit_x: deg    range:    24.062698 deg:   24.063349 deg
 n_y:      3  type_y: DEC--SIN  unit_y: deg    range:    29.934094 deg:   29.935209 deg
 n_s:      4  type_s: VOPT      unit_s: m / s  range:  -321214.699 m / s: -317350.054 m / s
        """.strip()


@pytest.mark.skipif(not ytOK, reason='Could not import yt')
class TestYt():
    def setup_method(self, method):
        self.cube = SpectralCube.read(path('adv.fits'))
        # Without any special arguments
        self.ytc1 = self.cube.to_yt()
        # With spectral factor = 0.5
        self.spectral_factor = 0.5
        self.ytc2 = self.cube.to_yt(spectral_factor=self.spectral_factor)
        # With nprocs = 4
        self.nprocs = 4
        self.ytc3 = self.cube.to_yt(nprocs=self.nprocs)

    def test_yt(self):
        # The following assertions just make sure everything is
        # kosher with the datasets generated in different ways
        ytc1,ytc2,ytc3 = self.ytc1,self.ytc2,self.ytc3
        ds1,ds2,ds3 = ytc1.dataset, ytc2.dataset, ytc3.dataset
        assert_array_equal(ds1.domain_dimensions, ds2.domain_dimensions)
        assert_array_equal(ds2.domain_dimensions, ds3.domain_dimensions)
        assert_allclose(ds1.domain_left_edge, ds2.domain_left_edge)
        assert_allclose(ds2.domain_left_edge, ds3.domain_left_edge)
        assert_allclose(ds1.domain_width, ds2.domain_width*np.array([1,1,1.0/self.spectral_factor]))
        assert_allclose(ds1.domain_width, ds3.domain_width)
        assert self.nprocs == len(ds3.index.grids)
        assert ds1.spec_cube
        assert ds2.spec_cube
        assert ds3.spec_cube
        ds1.index
        ds2.index
        ds3.index
        unit1 = ds1.field_info["fits","flux"].units
        unit2 = ds2.field_info["fits","flux"].units
        unit3 = ds3.field_info["fits","flux"].units
        ds1.quan(1.0,unit1)
        ds2.quan(1.0,unit2)
        ds3.quan(1.0,unit3)

    @pytest.mark.skipif(yt_version < StrictVersion('3.0.1'), reason='yt 3.0 has a FITS-related bug')
    def test_yt_fluxcompare(self):
        # Now check that we can compute quantities of the flux
        # and that they are equal
        ytc1,ytc2,ytc3 = self.ytc1,self.ytc2,self.ytc3
        ds1,ds2,ds3 = ytc1.dataset, ytc2.dataset, ytc3.dataset
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

    def test_yt_roundtrip_wcs(self):
        # Now test round-trip conversions between yt and world coordinates
        ytc1,ytc2,ytc3 = self.ytc1,self.ytc2,self.ytc3
        ds1,ds2,ds3 = ytc1.dataset, ytc2.dataset, ytc3.dataset
        cube = self.cube
        yt_coord1 = ds1.domain_left_edge + np.random.random(size=3)*ds1.domain_width
        world_coord1 = ytc1.yt2world(yt_coord1)
        assert_allclose(ytc1.world2yt(world_coord1), yt_coord1)
        yt_coord2 = ds2.domain_left_edge + np.random.random(size=3)*ds2.domain_width
        world_coord2 = ytc2.yt2world(yt_coord2)
        assert_allclose(ytc2.world2yt(world_coord2), yt_coord2)
        yt_coord3 = ds3.domain_left_edge + np.random.random(size=3)*ds3.domain_width
        world_coord3 = ytc3.yt2world(yt_coord3)
        assert_allclose(ytc3.world2yt(world_coord3), yt_coord3)

def test_read_write_rountrip(tmpdir):
    cube = SpectralCube.read(path('adv.fits'))
    tmp_file = str(tmpdir.join('test.fits'))
    cube.write(tmp_file)
    cube2 = SpectralCube.read(tmp_file)

    assert cube.shape == cube.shape
    assert_allclose(cube._data, cube2._data)
    if ((hasattr(_wcs, '__version__') and StrictVersion(_wcs.__version__) != StrictVersion('5.9'))
        or not hasattr(_wcs, '__version__')):
        # see https://github.com/astropy/astropy/pull/3992 for reasons:
        # we should upgrade this for 5.10 when the absolute accuracy is
        # maximized
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
    assert exc.value.args[0] == ("Mask shape is not broadcastable to data shape: "
                                 "(5, 5) vs (1, 1, 5)")


class TestMasks(BaseTest):

    @pytest.mark.parametrize('op', (operator.gt, operator.lt,
                             operator.le, operator.ge))
    def test_operator_threshold(self, op):

        # choose thresh to exercise proper equality tests
        thresh = self.d.ravel()[0]
        m = op(self.c, thresh*u.K)
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


@pytest.mark.skipif(not bottleneckOK, reason='Bottleneck could not be imported')
def test_endians():
    """
    Test that the endianness checking returns something in Native form
    (this is only needed for non-numpy functions that worry about the
    endianness of their data)

    WARNING: Because the endianness is machine-dependent, this may fail on
    different architectures!  This is because numpy automatically converts
    little-endian to native in the dtype parameter; I need a workaround for
    this.
    """
    big = np.array([[[1],[2]]], dtype='>f4')
    lil = np.array([[[1],[2]]], dtype='<f4')
    mywcs = WCS(naxis=3)
    mywcs.wcs.ctype[0] = 'RA'
    mywcs.wcs.ctype[1] = 'DEC'
    mywcs.wcs.ctype[2] = 'VELO'
    
    bigcube = SpectralCube(data=big, wcs=mywcs)
    xbig = bigcube._get_filled_data(check_endian=True)

    lilcube = SpectralCube(data=lil, wcs=mywcs)
    xlil = lilcube._get_filled_data(check_endian=True)

    assert xbig.dtype.byteorder == '='
    assert xlil.dtype.byteorder == '='

    xbig = bigcube._get_filled_data(check_endian=False)
    xlil = lilcube._get_filled_data(check_endian=False)

    assert xbig.dtype.byteorder == '>'
    assert xlil.dtype.byteorder == '='

def test_header_naxis():

    cube, data = cube_and_raw('advs.fits')

    assert cube.header['NAXIS'] == 3 # NOT data.ndim == 4
    assert cube.header['NAXIS1'] == data.shape[3]
    assert cube.header['NAXIS2'] == data.shape[2]
    assert cube.header['NAXIS3'] == data.shape[1]
    assert 'NAXIS4' not in cube.header

def test_slicing():

    cube, data = cube_and_raw('advs.fits')

    # just to check that we're starting in the right place
    assert cube.shape == (2,3,4)

    sl = cube[:,1,:]
    assert sl.shape == (2,4)
    
    v = cube[1:2,:,:]
    assert v.shape == (1,3,4)

    assert cube[:,:,:].shape == (2,3,4)
    assert cube[:,:].shape == (2,3,4)
    assert cube[:].shape == (2,3,4)
    assert cube[:1,:1,:1].shape == (1,1,1)


@pytest.mark.parametrize(('view','naxis'),
                         [( (slice(None), 1, slice(None)), 2 ),
                          ( (1, slice(None), slice(None)), 2 ),
                          ( (slice(None), slice(None), 1), 2 ),
                          ( (slice(None), slice(None), slice(1)), 3 ),
                          ( (slice(1), slice(1), slice(1)), 3 ),
                         ])
def test_slice_wcs(view, naxis):

    cube, data = cube_and_raw('advs.fits')

    sl = cube[view]
    assert sl.wcs.naxis == naxis

def test_header_units_consistent():

    cube, data = cube_and_raw('advs.fits')

    cube_kms = cube.with_spectral_unit(u.km/u.s)
    cube_Mms = cube.with_spectral_unit(u.Mm/u.s)

    assert cube.header['CUNIT3'] == 'm s-1'
    assert cube_kms.header['CUNIT3'] == 'km s-1'
    assert cube_Mms.header['CUNIT3'] == 'Mm s-1'

    # Wow, the tolerance here is really terrible...
    assert_allclose(cube_Mms.header['CDELT3'], cube.header['CDELT3']/1e6,rtol=1e-3,atol=1e-5)
    assert_allclose(cube.header['CDELT3']/1e3, cube_kms.header['CDELT3'],rtol=1e-2,atol=1e-5)

    cube_freq = cube.with_spectral_unit(u.Hz)

    assert cube_freq.header['CUNIT3'] == 'Hz'

    cube_freq_GHz = cube.with_spectral_unit(u.GHz)

    assert cube_freq_GHz.header['CUNIT3'] == 'GHz'

def test_spectral_unit_conventions():

    cube, data = cube_and_raw('advs.fits')
    cube_frq = cube.with_spectral_unit(u.Hz)

    cube_opt = cube.with_spectral_unit(u.km/u.s,
                                       rest_value=cube_frq.spectral_axis[0],
                                       velocity_convention='optical')
    cube_rad = cube.with_spectral_unit(u.km/u.s,
                                       rest_value=cube_frq.spectral_axis[0],
                                       velocity_convention='radio')
    cube_rel = cube.with_spectral_unit(u.km/u.s,
                                       rest_value=cube_frq.spectral_axis[0],
                                       velocity_convention='relativistic')

    # should all be exactly 0 km/s
    for x in (cube_rel.spectral_axis[0], cube_rad.spectral_axis[0],
              cube_opt.spectral_axis[0]):
        np.testing.assert_almost_equal(0,x.value)
    assert cube_rel.spectral_axis[1] != cube_rad.spectral_axis[1]
    assert cube_opt.spectral_axis[1] != cube_rad.spectral_axis[1]
    assert cube_rel.spectral_axis[1] != cube_opt.spectral_axis[1]

    assert cube_rel.velocity_convention == u.doppler_relativistic
    assert cube_rad.velocity_convention == u.doppler_radio
    assert cube_opt.velocity_convention == u.doppler_optical

def test_invalid_spectral_unit_conventions():

    cube, data = cube_and_raw('advs.fits')

    with pytest.raises(ValueError) as exc:
        cube.with_spectral_unit(u.km/u.s,
                                velocity_convention='invalid velocity convention')
    assert exc.value.args[0] == ("Velocity convention must be radio, optical, "
                                 "or relativistic.")

@pytest.mark.parametrize('rest', (50, 50*u.K))
def test_invalid_rest(rest):

    cube, data = cube_and_raw('advs.fits')

    with pytest.raises(ValueError) as exc:
        cube.with_spectral_unit(u.km/u.s,
                                velocity_convention='radio',
                                rest_value=rest)
    assert exc.value.args[0] == ("Rest value must be specified as an astropy "
                                 "quantity with spectral equivalence.")

def test_airwave_to_wave():

    cube, data = cube_and_raw('advs.fits')
    cube._wcs.wcs.ctype[2] = 'AWAV'
    cube._wcs.wcs.cunit[2] = 'm'
    cube._spectral_unit = u.m
    cube._wcs.wcs.cdelt[2] = 1e-7
    cube._wcs.wcs.crval[2] = 5e-7

    ax1 = cube.spectral_axis
    ax2 = cube.with_spectral_unit(u.m).spectral_axis
    np.testing.assert_almost_equal(spectral_axis.air_to_vac(ax1).value,
                                   ax2.value)

@pytest.mark.parametrize(('func','how','axis'),
                         itertools.product(('sum','std','max','min','mean'),
                                           ('slice','cube','auto'),
                                           (0,1,2)
                                          ))
def test_twod_numpy(func, how, axis):
    # Check that a numpy function returns the correct result when applied along
    # one axis
    # This is partly a regression test for #211

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K

    proj = getattr(cube,func)(axis=axis, how=how)
    # data has a redundant 1st axis
    dproj = getattr(data,func)(axis=(0,axis+1)).squeeze()
    assert isinstance(proj, Projection)
    np.testing.assert_equal(proj.value, dproj)
    assert cube.unit == proj.unit

@pytest.mark.parametrize('func',('sum','std','max','min','mean'))
def test_oned_numpy(func):
    # Check that a numpy function returns an appropriate spectrum

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K

    spec = getattr(cube,func)(axis=(1,2))
    dspec = getattr(data,func)(axis=(2,3)).squeeze()
    assert isinstance(spec, OneDSpectrum)
    # data has a redundant 1st axis
    np.testing.assert_equal(spec.value, dspec)
    assert cube.unit == spec.unit

def test_oned_slice():
    # Check that a slice returns an appropriate spectrum

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K

    spec = cube[:,0,0]
    assert isinstance(spec, OneDSpectrum)
    # data has a redundant 1st axis
    np.testing.assert_equal(spec.value, data[0,:,0,0])
    assert cube.unit == spec.unit

def test_preserve_bunit():

    cube, data = cube_and_raw('advs.fits')

    assert cube.header['BUNIT'] == 'K'

    hdu = fits.open(path('advs.fits'))[0]
    hdu.header['BUNIT'] = 'Jy'
    cube = SpectralCube.read(hdu)

    assert cube.unit == u.Jy
    assert cube.header['BUNIT'] == 'Jy'

def test_cube_with_swapped_axes():
    """
    Regression test for #208
    """
    cube, data = cube_and_raw('vda.fits')

    # Check that masking works (this should apply a lazy mask)
    cube.filled_data[:]

def test_jybeam_upper():

    cube, data = cube_and_raw('vda_JYBEAM_upper.fits')

    assert cube.unit == u.Jy
    if RADIO_BEAM_INSTALLED:
        assert hasattr(cube, 'beam')
        np.testing.assert_almost_equal(cube.beam.sr.value,
                                       (((1*u.arcsec/np.sqrt(8*np.log(2)))**2).to(u.sr)*2*np.pi).value)

def test_jybeam_lower():

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    assert cube.unit == u.Jy
    if RADIO_BEAM_INSTALLED:
        assert hasattr(cube, 'beam')
        np.testing.assert_almost_equal(cube.beam.sr.value,
                                       (((1*u.arcsec/np.sqrt(8*np.log(2)))**2).to(u.sr)*2*np.pi).value)
