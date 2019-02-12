from __future__ import print_function, absolute_import, division

import operator
import itertools
import warnings
import mmap
from distutils.version import LooseVersion

import pytest

import astropy
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs import _wcs
from astropy.tests.helper import assert_quantity_allclose
from astropy.extern import six
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel
import numpy as np

from .. import (SpectralCube, VaryingResolutionSpectralCube, BooleanArrayMask,
                FunctionMask, LazyMask, CompositeMask)
from ..spectral_cube import (OneDSpectrum, Projection,
                             VaryingResolutionOneDSpectrum,
                             LowerDimensionalObject)
from ..np_compat import allbadtonan
from .. import spectral_axis
from .. import base_class
from .. import utils

from . import path
from .helpers import assert_allclose, assert_array_equal

# needed to test for warnings later
warnings.simplefilter('always', UserWarning)
warnings.simplefilter('error', utils.UnsupportedIterationStrategyWarning)
warnings.simplefilter('error', utils.NotImplementedWarning)
warnings.simplefilter('error', utils.WCSMismatchWarning)


try:
    import yt
    YT_INSTALLED = True
    YT_LT_301 = LooseVersion(yt.__version__) < LooseVersion('3.0.1')
except ImportError:
    YT_INSTALLED = False
    YT_LT_301 = False

from radio_beam import Beam, Beams

NUMPY_LT_19 = LooseVersion(np.__version__) < LooseVersion('1.9.0')


def cube_and_raw(filename):
    p = path(filename)

    d = fits.getdata(p)

    c = SpectralCube.read(p, format='fits', mode='readonly')
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

    cube = SpectralCube(data=data, wcs=cube.wcs)

    assert not cube._is_huge

    # We need to reduce the memory threshold rather than use a large cube to
    # make sure we don't use too much memory during testing.
    from .. import cube_utils
    OLD_MEMORY_THRESHOLD = cube_utils.MEMORY_THRESHOLD

    try:
        cube_utils.MEMORY_THRESHOLD = 10

        assert cube._is_huge

        with pytest.raises(ValueError) as exc:
            cube + 5*cube.unit
        assert 'entire cube into memory' in exc.value.args[0]

        with pytest.raises(ValueError) as exc:
            cube.max(how='cube')
        assert 'entire cube into memory' in exc.value.args[0]


        cube.allow_huge_operations = True

        # just make sure it doesn't fail
        cube + 5*cube.unit
    finally:
        cube_utils.MEMORY_THRESHOLD = OLD_MEMORY_THRESHOLD


class BaseTest(object):

    def setup_method(self, method):
        c, d = cube_and_raw('adv.fits')
        mask = BooleanArrayMask(d > 0.5, c._wcs)
        c._mask = mask
        self.c = c
        self.mask = mask
        self.d = d

class BaseTestMultiBeams(object):

    def setup_method(self, method):
        c, d = cube_and_raw('adv_beams.fits')
        mask = BooleanArrayMask(d > 0.5, c._wcs)
        c._mask = mask
        self.c = c
        self.mask = mask
        self.d = d

translist = [('advs', [0, 1, 2, 3]),
             ('dvsa', [2, 3, 0, 1]),
             ('sdav', [0, 2, 1, 3]),
             ('sadv', [0, 1, 2, 3]),
             ('vsad', [3, 0, 1, 2]),
             ('vad', [2, 0, 1]),
             ('vda', [0, 2, 1]),
             ('adv', [0, 1, 2]),
             ]

translist_vrsc = [('vda_beams', [0, 2, 1])]

class TestSpectralCube(object):

    @pytest.mark.parametrize(('name', 'trans'), translist + translist_vrsc)
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


    @pytest.mark.parametrize(('name','masktype','unit'),
                             itertools.product(('advs', 'dvsa', 'sdav', 'sadv', 'vsad', 'vad', 'adv',),
                                               (BooleanArrayMask, LazyMask, FunctionMask, CompositeMask),
                                               ('Hz', u.Hz),
                                              )
                            )
    def test_with_spectral_unit(self, name, masktype, unit):
        cube, data = cube_and_raw(name + '.fits')
        cube_freq = cube.with_spectral_unit(unit)

        if masktype == BooleanArrayMask:
            # don't use data here:
            # data haven't necessarily been rearranged to the correct shape by
            # cube_utils.orient
            mask = BooleanArrayMask(cube.filled_data[:].value>0,
                                    wcs=cube._wcs)
        elif masktype == LazyMask:
            mask = LazyMask(lambda x: x>0, cube=cube)
        elif masktype == FunctionMask:
            mask = FunctionMask(lambda x: x>0)
        elif masktype == CompositeMask:
            mask1 = FunctionMask(lambda x: x>0)
            mask2 = LazyMask(lambda x: x>0, cube)
            mask = CompositeMask(mask1, mask2)

        cube2 = cube.with_mask(mask)
        cube_masked_freq = cube2.with_spectral_unit(unit)

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

    @pytest.mark.parametrize(('name', 'trans'), translist)
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

    @pytest.mark.parametrize(('name', 'trans'), translist_vrsc)
    def test_getitem_vrsc(self, name, trans):
        c, d = cube_and_raw(name + '.fits')

        expected = np.squeeze(d.transpose(trans))

        # No pv slices for VRSC.

        assert_allclose(c[0,:,:].value, expected[0,:,:])

        # Not implemented:
        #assert_allclose(c[0,0,:].value, expected[0,0,:])
        #assert_allclose(c[0,:,0].value, expected[0,:,0])
        assert_allclose(c[:,0,0].value, expected[:,0,0])

        assert_allclose(c[1,:,:].value, expected[1,:,:])

        # Not implemented:
        #assert_allclose(c[1,1,:].value, expected[1,1,:])
        #assert_allclose(c[1,:,1].value, expected[1,:,1])
        assert_allclose(c[:,1,1].value, expected[:,1,1])

        c2 = c.with_spectral_unit(u.km/u.s, velocity_convention='radio')

        assert_allclose(c2[0,:,:].value, expected[0,:,:])

        # Not implemented:
        #assert_allclose(c2[0,0,:].value, expected[0,0,:])
        #assert_allclose(c2[0,:,0].value, expected[0,:,0])
        assert_allclose(c2[:,0,0].value, expected[:,0,0])

        assert_allclose(c2[1,:,:].value, expected[1,:,:])

        # Not implemented:
        #assert_allclose(c2[1,1,:].value, expected[1,1,:])
        #assert_allclose(c2[1,:,1].value, expected[1,:,1])
        assert_allclose(c2[:,1,1].value, expected[:,1,1])

        # @pytest.mark.xfail(raises=AttributeError)
        @pytest.mark.parametrize(('name', 'trans'), translist_vrsc)
        def test_getitem_vrsc(self, name, trans):
            c, d = cube_and_raw(name + '.fits')

            expected = np.squeeze(d.transpose(trans))

            assert_allclose(c[:,:,0].value, expected[:,:,0])


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

        # regression test #251: the _data attribute must not be a quantity
        assert not hasattr(c2._data, 'unit')

    def test_subtract_cubes(self):
        d2 = self.d1 - self.d1
        c2 = self.c1 - self.c1
        assert np.all(d2 == c2.filled_data[:].value)
        assert np.all(c2.filled_data[:].value == 0)
        assert c2.unit == u.K

        # regression test #251: the _data attribute must not be a quantity
        assert not hasattr(c2._data, 'unit')

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

    @pytest.mark.parametrize('iterate_rays', (True,False))
    def test_median(self, iterate_rays):
        # Make sure that medians ignore empty/bad/NaN values
        m = np.empty(self.d.shape[1:])
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                ray = self.d[:, y, x]
                # the cube mask is for values >0.5
                ray = ray[ray > 0.5]
                m[y, x] = np.median(ray)
        scmed = self.c.median(axis=0, iterate_rays=iterate_rays)
        assert_allclose(scmed, m)
        assert not np.any(np.isnan(scmed.value))
        assert scmed.unit == self.c.unit

    @pytest.mark.skipif('NUMPY_LT_19')
    def test_bad_median_apply(self):
        # this is a test for manually-applied numpy medians, which are different
        # from the cube.median method that does "the right thing"
        #
        # for regular median, we expect a failure, which is why we don't use
        # regular median.

        scmed = self.c.apply_numpy_function(np.median, axis=0)
        # this checks whether numpy <=1.9.3 has a bug?
        # as far as I can tell, np==1.9.3 no longer has this bug/feature
        #if LooseVersion(np.__version__) <= LooseVersion('1.9.3'):
        #    # print statements added so we get more info in the travis builds
        #    print("Numpy version is: {0}".format(LooseVersion(np.__version__)))
        #    assert np.count_nonzero(np.isnan(scmed)) == 5
        #else:
        #    print("Numpy version is: {0}".format(LooseVersion(np.__version__)))
        assert np.count_nonzero(np.isnan(scmed)) == 6

        scmed = self.c.apply_numpy_function(np.nanmedian, axis=0)
        assert np.count_nonzero(np.isnan(scmed)) == 0

        # use a more aggressive mask to force there to be some all-nan axes
        m2 = self.c>0.65*self.c.unit
        scmed = self.c.with_mask(m2).apply_numpy_function(np.nanmedian, axis=0)
        assert np.count_nonzero(np.isnan(scmed)) == 1

    @pytest.mark.parametrize('iterate_rays', (True,False))
    def test_bad_median(self, iterate_rays):
        # This should have the same result as np.nanmedian, though it might be
        # faster if bottleneck loads
        scmed = self.c.median(axis=0, iterate_rays=iterate_rays)
        assert np.count_nonzero(np.isnan(scmed)) == 0

        m2 = self.c>0.65*self.c.unit
        scmed = self.c.with_mask(m2).median(axis=0, iterate_rays=iterate_rays)
        assert np.count_nonzero(np.isnan(scmed)) == 1

    @pytest.mark.parametrize(('pct', 'iterate_rays'),
                             (zip((3,25,50,75,97)*2,(True,)*5 + (False,)*5)))
    def test_percentile(self, pct, iterate_rays):
        m = np.empty(self.d.sum(axis=0).shape)
        for y in range(m.shape[0]):
            for x in range(m.shape[1]):
                ray = self.d[:, y, x]
                ray = ray[ray > 0.5]
                m[y, x] = np.percentile(ray, pct)
        scpct = self.c.percentile(pct, axis=0, iterate_rays=iterate_rays)
        assert_allclose(scpct, m)
        assert not np.any(np.isnan(scpct.value))
        assert scpct.unit == self.c.unit

    @pytest.mark.parametrize('method', ('sum', 'min', 'max', 'std', 'mad_std',
                                        'median', 'argmin', 'argmax'))
    def test_transpose(self, method):
        c1, d1 = cube_and_raw('adv.fits')
        c2, d2 = cube_and_raw('vad.fits')
        for axis in [None, 0, 1, 2]:
            assert_allclose(getattr(c1, method)(axis=axis),
                            getattr(c2, method)(axis=axis))
            # check that all these accept progressbar kwargs
            assert_allclose(getattr(c1, method)(axis=axis, progressbar=True),
                            getattr(c2, method)(axis=axis, progressbar=True))


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

class TestSlabMultiBeams(BaseTestMultiBeams, TestSlab):
    """ same tests with multibeams """
    pass


class TestRepr(BaseTest):

    def test_repr(self):
        assert repr(self.c) == """
SpectralCube with shape=(4, 3, 2) and unit=K:
 n_x:      2  type_x: RA---SIN  unit_x: deg    range:    24.062698 deg:   24.063349 deg
 n_y:      3  type_y: DEC--SIN  unit_y: deg    range:    29.934094 deg:   29.935209 deg
 n_s:      4  type_s: VOPT      unit_s: km / s  range:     -321.215 km / s:    -317.350 km / s
        """.strip()

    def test_repr_withunit(self):
        self.c._unit = u.Jy
        assert repr(self.c) == """
SpectralCube with shape=(4, 3, 2) and unit=Jy:
 n_x:      2  type_x: RA---SIN  unit_x: deg    range:    24.062698 deg:   24.063349 deg
 n_y:      3  type_y: DEC--SIN  unit_y: deg    range:    29.934094 deg:   29.935209 deg
 n_s:      4  type_s: VOPT      unit_s: km / s  range:     -321.215 km / s:    -317.350 km / s
        """.strip()


@pytest.mark.skipif('not YT_INSTALLED')
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
        assert_allclose(ds1.domain_left_edge.value, ds2.domain_left_edge.value)
        assert_allclose(ds2.domain_left_edge.value, ds3.domain_left_edge.value)
        assert_allclose(ds1.domain_width.value,
                        ds2.domain_width.value*np.array([1,1,1.0/self.spectral_factor]))
        assert_allclose(ds1.domain_width.value, ds3.domain_width.value)
        assert self.nprocs == len(ds3.index.grids)

        ds1.index
        ds2.index
        ds3.index
        unit1 = ds1.field_info["fits","flux"].units
        unit2 = ds2.field_info["fits","flux"].units
        unit3 = ds3.field_info["fits","flux"].units
        ds1.quan(1.0,unit1)
        ds2.quan(1.0,unit2)
        ds3.quan(1.0,unit3)

    @pytest.mark.skipif('YT_LT_301', reason='yt 3.0 has a FITS-related bug')
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
        yt_coord1 = ds1.domain_left_edge + np.random.random(size=3)*ds1.domain_width
        world_coord1 = ytc1.yt2world(yt_coord1)
        assert_allclose(ytc1.world2yt(world_coord1), yt_coord1.value)
        yt_coord2 = ds2.domain_left_edge + np.random.random(size=3)*ds2.domain_width
        world_coord2 = ytc2.yt2world(yt_coord2)
        assert_allclose(ytc2.world2yt(world_coord2), yt_coord2.value)
        yt_coord3 = ds3.domain_left_edge + np.random.random(size=3)*ds3.domain_width
        world_coord3 = ytc3.yt2world(yt_coord3)
        assert_allclose(ytc3.world2yt(world_coord3), yt_coord3.value)

def test_read_write_rountrip(tmpdir):
    cube = SpectralCube.read(path('adv.fits'))
    tmp_file = str(tmpdir.join('test.fits'))
    cube.write(tmp_file)
    cube2 = SpectralCube.read(tmp_file)

    assert cube.shape == cube.shape
    assert_allclose(cube._data, cube2._data)
    if (((hasattr(_wcs, '__version__')
          and LooseVersion(_wcs.__version__) < LooseVersion('5.9'))
         or not hasattr(_wcs, '__version__'))):
        # see https://github.com/astropy/astropy/pull/3992 for reasons:
        # we should upgrade this for 5.10 when the absolute accuracy is
        # maximized
        assert cube._wcs.to_header_string() == cube2._wcs.to_header_string()
        # in 5.11 and maybe even 5.12, the round trip fails.  Maybe
        # https://github.com/astropy/astropy/issues/4292 will solve it?

@pytest.mark.parametrize(('memmap', 'base'),
                         ((True, mmap.mmap),
                          (False, None)))
def test_read_memmap(memmap, base):
    cube = SpectralCube.read(path('adv.fits'), memmap=memmap)

    bb = cube.base
    while hasattr(bb, 'base'):
        bb = bb.base

    if base is None:
        assert bb is None
    else:
        assert isinstance(bb, base)


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


def test_with_mask_with_good_array_shape():
    cube = _dummy_cube()
    mask = np.zeros((1, 5), dtype=np.bool)
    cube2 = cube.with_mask(mask, inherit_mask=False)
    assert isinstance(cube2._mask, BooleanArrayMask)
    np.testing.assert_equal(cube2._mask._mask, mask.reshape((1, 1, 5)))


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
    pytest.importorskip('bottleneck')
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

    # make sure this works.  Not sure what keys to test for...
    v.header

    assert cube[:,:,:].shape == (2,3,4)
    assert cube[:,:].shape == (2,3,4)
    assert cube[:].shape == (2,3,4)
    assert cube[:1,:1,:1].shape == (1,1,1)


@pytest.mark.parametrize(('view','naxis'),
                         [((slice(None), 1, slice(None)), 2),
                          ((1, slice(None), slice(None)), 2),
                          ((slice(None), slice(None), 1), 2),
                          ((slice(None), slice(None), slice(1)), 3),
                          ((slice(1), slice(1), slice(1)), 3),
                          ((slice(None, None, -1), slice(None), slice(None)), 3),
                         ])
def test_slice_wcs(view, naxis):

    cube, data = cube_and_raw('advs.fits')

    sl = cube[view]
    assert sl.wcs.naxis == naxis

def test_slice_wcs_reversal():
    cube, data = cube_and_raw('advs.fits')
    view = (slice(None,None,-1), slice(None), slice(None))

    rcube = cube[view]
    rrcube = rcube[view]

    np.testing.assert_array_equal(np.diff(cube.spectral_axis),
                                  -np.diff(rcube.spectral_axis))

    np.testing.assert_array_equal(rrcube.spectral_axis.value,
                                  cube.spectral_axis.value)
    np.testing.assert_array_equal(rcube.spectral_axis.value,
                                  cube.spectral_axis.value[::-1])
    np.testing.assert_array_equal(rrcube.world_extrema.value,
                                  cube.world_extrema.value)
    # check that the lon, lat arrays are *entirely* unchanged
    np.testing.assert_array_equal(rrcube.spatial_coordinate_map[0].value,
                                  cube.spatial_coordinate_map[0].value)
    np.testing.assert_array_equal(rrcube.spatial_coordinate_map[1].value,
                                  cube.spatial_coordinate_map[1].value)

def test_spectral_slice_preserve_units():
    cube, data = cube_and_raw('advs.fits')
    cube = cube.with_spectral_unit(u.km/u.s)

    sl = cube[:,0,0]

    assert cube._spectral_unit == u.km/u.s
    assert sl._spectral_unit == u.km/u.s

    assert cube.spectral_axis.unit == u.km/u.s
    assert sl.spectral_axis.unit == u.km/u.s

def test_header_units_consistent():

    cube, data = cube_and_raw('advs.fits')

    cube_ms = cube.with_spectral_unit(u.m/u.s)
    cube_kms = cube.with_spectral_unit(u.km/u.s)
    cube_Mms = cube.with_spectral_unit(u.Mm/u.s)

    assert cube.header['CUNIT3'] == 'km s-1'
    assert cube_ms.header['CUNIT3'] == 'm s-1'
    assert cube_kms.header['CUNIT3'] == 'km s-1'
    assert cube_Mms.header['CUNIT3'] == 'Mm s-1'

    # Wow, the tolerance here is really terrible...
    assert_allclose(cube_Mms.header['CDELT3'], cube.header['CDELT3']/1e3,rtol=1e-3,atol=1e-5)
    assert_allclose(cube.header['CDELT3'], cube_kms.header['CDELT3'],rtol=1e-2,atol=1e-5)
    assert_allclose(cube.header['CDELT3']*1e3, cube_ms.header['CDELT3'],rtol=1e-2,atol=1e-5)

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

@pytest.mark.parametrize(('func','how','axis'),
                         itertools.product(('sum','std','max','min','mean'),
                                           ('slice','cube','auto'),
                                           ((0,1),(1,2),(0,2))
                                          ))
def test_twod_numpy_twoaxes(func, how, axis):
    # Check that a numpy function returns the correct result when applied along
    # one axis
    # This is partly a regression test for #211

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K

    if func == 'mean' and axis != (1,2):
        with warnings.catch_warnings(record=True) as wrn:
            spec = getattr(cube,func)(axis=axis, how=how)

        assert 'Averaging over a spatial and a spectral' in str(wrn[-1].message)

    spec = getattr(cube,func)(axis=axis, how=how)
    # data has a redundant 1st axis
    dspec = getattr(data.squeeze(),func)(axis=axis)

    if axis == (1,2):
        assert isinstance(spec, OneDSpectrum)
        assert cube.unit == spec.unit
        np.testing.assert_almost_equal(spec.value, dspec)
    else:
        np.testing.assert_almost_equal(spec, dspec)

def test_preserves_header_values():
    # Check that the non-WCS header parameters are preserved during projection

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K
    cube._header['OBJECT'] = 'TestName'

    proj = cube.sum(axis=0, how='auto')
    assert isinstance(proj, Projection)
    assert proj.header['OBJECT'] == 'TestName'
    assert proj.hdu.header['OBJECT'] == 'TestName'

def test_preserves_header_meta_values():
    # Check that additional parameters in meta are preserved

    cube, data = cube_and_raw('advs.fits')

    cube.meta['foo'] = 'bar'

    assert cube.header['FOO'] == 'bar'

    # check that long keywords are also preserved
    cube.meta['too_long_keyword'] = 'too_long_information'

    assert 'too_long_keyword=too_long_information' in cube.header['COMMENT']

    # Checks that the header is preserved when passed to LDOs
    for ldo in (cube.sum(axis=0, how='auto'), cube[:,0,0]):
        assert isinstance(ldo, LowerDimensionalObject)
        assert ldo.header['FOO'] == 'bar'
        assert ldo.hdu.header['FOO'] == 'bar'

        # make sure that the meta preservation works on the LDOs themselves too
        ldo.meta['bar'] = 'foo'
        assert ldo.header['BAR'] == 'foo'

        assert 'too_long_keyword=too_long_information' in ldo.header['COMMENT']



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
    assert spec.header['BUNIT'] == cube.header['BUNIT']


def test_oned_slice_beams():
    # Check that a slice returns an appropriate spectrum

    cube, data = cube_and_raw('sdav_beams.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K

    spec = cube[:,0,0]
    assert isinstance(spec, VaryingResolutionOneDSpectrum)
    # data has a redundant 1st axis
    np.testing.assert_equal(spec.value, data[:,0,0,0])
    assert cube.unit == spec.unit
    assert spec.header['BUNIT'] == cube.header['BUNIT']

    assert hasattr(spec, 'beams')
    assert 'BMAJ' in spec.hdulist[1].data.names

def test_subcube_slab_beams():
    cube, data = cube_and_raw('sdav_beams.fits')

    slcube = cube[1:]

    assert all(slcube.hdulist[1].data['CHAN'] == np.arange(slcube.shape[0]))

    try:
        # Make sure Beams has been sliced correctly
        assert all(cube.beams[1:] == slcube.beams)
    except TypeError:
        # in 69eac9241220d3552c06b173944cb7cdebeb47ef, radio_beam switched to
        # returning a single value
        assert cube.beams[1:] == slcube.beams

# collapsing to one dimension raywise doesn't make sense and is therefore
# not supported.
@pytest.mark.parametrize('how', ('auto', 'cube', 'slice'))
def test_oned_collapse(how):
    # Check that an operation along the spatial dims returns an appropriate
    # spectrum

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K

    spec = cube.mean(axis=(1,2), how=how)
    assert isinstance(spec, OneDSpectrum)
    # data has a redundant 1st axis
    np.testing.assert_equal(spec.value, data.mean(axis=(0,2,3)))
    assert cube.unit == spec.unit
    assert spec.header['BUNIT'] == cube.header['BUNIT']


def test_oned_collapse_beams():
    # Check that an operation along the spatial dims returns an appropriate
    # spectrum

    cube, data = cube_and_raw('sdav_beams.fits')
    cube._meta['BUNIT'] = 'K'
    cube._unit = u.K

    spec = cube.mean(axis=(1,2))
    assert isinstance(spec, VaryingResolutionOneDSpectrum)
    # data has a redundant 1st axis
    np.testing.assert_equal(spec.value, data.mean(axis=(1,2,3)))
    assert cube.unit == spec.unit
    assert spec.header['BUNIT'] == cube.header['BUNIT']

    assert hasattr(spec, 'beams')
    assert 'BMAJ' in spec.hdulist[1].data.names

def test_preserve_bunit():

    cube, data = cube_and_raw('advs.fits')

    assert cube.header['BUNIT'] == 'K'

    hdu = fits.open(path('advs.fits'))[0]
    hdu.header['BUNIT'] = 'Jy'
    cube = SpectralCube.read(hdu)

    assert cube.unit == u.Jy
    assert cube.header['BUNIT'] == 'Jy'


def test_preserve_beam():

    cube, data = cube_and_raw('advs.fits')

    beam = Beam.from_fits_header(path("advs.fits"))

    assert cube.beam == beam


def test_beam_attach_to_header():

    cube, data = cube_and_raw('adv.fits')

    header = cube._header.copy()
    del header["BMAJ"], header["BMIN"], header["BPA"]

    newcube = SpectralCube(data=data, wcs=cube.wcs, header=header,
                           beam=cube.beam)

    assert cube.header["BMAJ"] == newcube.header["BMAJ"]
    assert cube.header["BMIN"] == newcube.header["BMIN"]
    assert cube.header["BPA"] == newcube.header["BPA"]

    # Should be in meta too
    assert newcube.meta['beam'] == cube.beam


def test_beam_custom():

    cube, data = cube_and_raw('adv.fits')

    header = cube._header.copy()
    beam = Beam.from_fits_header(header)
    del header["BMAJ"], header["BMIN"], header["BPA"]

    newcube = SpectralCube(data=data, wcs=cube.wcs, header=header)

    # newcube should now not have a beam
    assert not hasattr(newcube, "beam")

    # Attach the beam
    newcube = newcube.with_beam(beam=beam)

    assert newcube.beam == cube.beam

    # Header should be updated
    assert cube.header["BMAJ"] == newcube.header["BMAJ"]
    assert cube.header["BMIN"] == newcube.header["BMIN"]
    assert cube.header["BPA"] == newcube.header["BPA"]

    # Should be in meta too
    assert newcube.meta['beam'] == cube.beam

    # Try changing the beam properties
    newbeam = Beam(beam.major * 2)

    newcube2 = newcube.with_beam(beam=newbeam)

    assert newcube2.beam == newbeam

    # Header should be updated
    assert newcube2.header["BMAJ"] == newbeam.major.value
    assert newcube2.header["BMIN"] == newbeam.minor.value
    assert newcube2.header["BPA"] == newbeam.pa.value

    # Should be in meta too
    assert newcube2.meta['beam'] == newbeam


def test_multibeam_custom():

    cube, data = cube_and_raw('vda_beams.fits')

    # Make a new set of beams that differs from the original.
    new_beams = Beams([1.] * cube.shape[0] * u.deg)

    # Attach the beam
    newcube = cube.with_beams(new_beams)

    try:
        assert all(new_beams == newcube.beams)
    except TypeError:
        # in 69eac9241220d3552c06b173944cb7cdebeb47ef, radio_beam switched to
        # returning a single value
        assert new_beams == newcube.beams


@pytest.mark.xfail(raises=ValueError, strict=True)
def test_multibeam_custom_wrongshape():

    cube, data = cube_and_raw('vda_beams.fits')

    # Make a new set of beams that differs from the original.
    new_beams = Beams([1.] * cube.shape[0] * u.deg)

    # Attach the beam
    cube.with_beams(new_beams[:1])


def test_multibeam_slice():

    cube, data = cube_and_raw('vda_beams.fits')

    assert isinstance(cube, VaryingResolutionSpectralCube)
    np.testing.assert_almost_equal(cube.beams[0].major.value, 0.1)
    np.testing.assert_almost_equal(cube.beams[3].major.value, 0.4)

    scube = cube[:2,:,:]

    np.testing.assert_almost_equal(scube.beams[0].major.value, 0.1)
    np.testing.assert_almost_equal(scube.beams[1].major.value, 0.2)

    flatslice = cube[0,:,:]

    np.testing.assert_almost_equal(flatslice.header['BMAJ'],
                                   (0.1/3600.))

    # Test returning a VRODS

    spec = cube[:, 0, 0]

    assert (cube.beams == spec.beams).all()

    # And make sure that Beams gets slice for part of a spectrum

    spec_part = cube[:1, 0, 0]

    assert cube.beams[0] == spec.beams[0]

def test_basic_unit_conversion():

    cube, data = cube_and_raw('advs.fits')
    assert cube.unit == u.K

    mKcube = cube.to(u.mK)

    np.testing.assert_almost_equal(mKcube.filled_data[:].value,
                                   (cube.filled_data[:].value *
                                    1e3))


def test_basic_unit_conversion_beams():
    cube, data = cube_and_raw('vda_beams.fits')
    cube._unit = u.K # want beams, but we want to force the unit to be something non-beamy
    cube._meta['BUNIT'] = 'K'

    assert cube.unit == u.K

    mKcube = cube.to(u.mK)

    np.testing.assert_almost_equal(mKcube.filled_data[:].value,
                                   (cube.filled_data[:].value *
                                    1e3))



def test_beam_jtok_array():

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'Jy / beam'
    cube._unit = u.Jy/u.beam

    equiv = cube.beam.jtok_equiv(cube.with_spectral_unit(u.GHz).spectral_axis)
    jtok = cube.beam.jtok(cube.with_spectral_unit(u.GHz).spectral_axis)

    Kcube = cube.to(u.K, equivalencies=equiv)
    np.testing.assert_almost_equal(Kcube.filled_data[:].value,
                                   (cube.filled_data[:].value *
                                    jtok[:,None,None]).value)

    # test that the beam equivalencies are correctly automatically defined
    Kcube = cube.to(u.K)
    np.testing.assert_almost_equal(Kcube.filled_data[:].value,
                                   (cube.filled_data[:].value *
                                    jtok[:,None,None]).value)


def test_multibeam_jtok_array():

    cube, data = cube_and_raw('vda_beams.fits')
    assert cube.meta['BUNIT'].strip() == 'Jy / beam'
    assert cube.unit.is_equivalent(u.Jy/u.beam)

    #equiv = [bm.jtok_equiv(frq) for bm, frq in zip(cube.beams, cube.with_spectral_unit(u.GHz).spectral_axis)]
    jtok = u.Quantity([bm.jtok(frq) for bm, frq in zip(cube.beams, cube.with_spectral_unit(u.GHz).spectral_axis)])

    # don't try this, it's nonsense for the multibeam case
    # Kcube = cube.to(u.K, equivalencies=equiv)
    # np.testing.assert_almost_equal(Kcube.filled_data[:].value,
    #                                (cube.filled_data[:].value *
    #                                 jtok[:,None,None]).value)

    # test that the beam equivalencies are correctly automatically defined
    Kcube = cube.to(u.K)
    np.testing.assert_almost_equal(Kcube.filled_data[:].value,
                                   (cube.filled_data[:].value *
                                    jtok[:,None,None]).value)



def test_beam_jtok():
    # regression test for an error introduced when the previous test was solved
    # (the "is this an array?" test used len(x) where x could be scalar)

    cube, data = cube_and_raw('advs.fits')
    # technically this should be jy/beam, but astropy's equivalency doesn't
    # handle this yet
    cube._meta['BUNIT'] = 'Jy'
    cube._unit = u.Jy

    equiv = cube.beam.jtok_equiv(np.median(cube.with_spectral_unit(u.GHz).spectral_axis))
    jtok = cube.beam.jtok(np.median(cube.with_spectral_unit(u.GHz).spectral_axis))

    Kcube = cube.to(u.K, equivalencies=equiv)
    np.testing.assert_almost_equal(Kcube.filled_data[:].value,
                                   (cube.filled_data[:].value *
                                    jtok).value)


def test_varyres_moment():
    cube, data = cube_and_raw('vda_beams.fits')

    assert isinstance(cube, VaryingResolutionSpectralCube)

    # the beams are very different, but for this test we don't care
    cube.beam_threshold = 1.0

    with warnings.catch_warnings(record=True) as wrn:
        warnings.simplefilter('default')
        m0 = cube.moment0()

    assert "Arithmetic beam averaging is being performed" in str(wrn[-1].message)
    assert_quantity_allclose(m0.meta['beam'].major, 0.25*u.arcsec)


def test_append_beam_to_hdr():

    cube, data = cube_and_raw('advs.fits')

    orig_hdr = fits.getheader(path('advs.fits'))

    assert cube.header['BMAJ'] == orig_hdr['BMAJ']
    assert cube.header['BMIN'] == orig_hdr['BMIN']
    assert cube.header['BPA'] == orig_hdr['BPA']

def test_cube_with_swapped_axes():
    """
    Regression test for #208
    """
    cube, data = cube_and_raw('vda.fits')

    # Check that masking works (this should apply a lazy mask)
    cube.filled_data[:]

def test_jybeam_upper():

    cube, data = cube_and_raw('vda_JYBEAM_upper.fits')

    assert cube.unit == u.Jy/u.beam
    assert hasattr(cube, 'beam')
    np.testing.assert_almost_equal(cube.beam.sr.value,
                                   (((1*u.arcsec/np.sqrt(8*np.log(2)))**2).to(u.sr)*2*np.pi).value)

def test_jybeam_lower():

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    assert cube.unit == u.Jy/u.beam
    assert hasattr(cube, 'beam')
    np.testing.assert_almost_equal(cube.beam.sr.value,
                                   (((1*u.arcsec/np.sqrt(8*np.log(2)))**2).to(u.sr)*2*np.pi).value)

# Regression test for #257 (https://github.com/radio-astro-tools/spectral-cube/pull/257)
def test_jybeam_whitespace():

    cube, data = cube_and_raw('vda_Jybeam_whitespace.fits')

    assert cube.unit == u.Jy/u.beam
    assert hasattr(cube, 'beam')
    np.testing.assert_almost_equal(cube.beam.sr.value,
                                   (((1*u.arcsec/np.sqrt(8*np.log(2)))**2).to(u.sr)*2*np.pi).value)


def test_beam_proj_meta():

    cube, data = cube_and_raw('advs.fits')

    moment = cube.moment0(axis=0)

    # regression test for #250
    assert 'beam' in moment.meta
    assert 'BMAJ' in moment.hdu.header

    slc = cube[0,:,:]

    assert 'beam' in slc.meta

    proj = cube.max(axis=0)

    assert 'beam' in proj.meta

def test_proj_meta():

    cube, data = cube_and_raw('advs.fits')

    moment = cube.moment0(axis=0)

    assert 'BUNIT' in moment.meta
    assert moment.meta['BUNIT'] == 'K'

    slc = cube[0,:,:]

    assert 'BUNIT' in slc.meta
    assert slc.meta['BUNIT'] == 'K'

    proj = cube.max(axis=0)

    assert 'BUNIT' in proj.meta
    assert proj.meta['BUNIT'] == 'K'

def test_pix_sign():

    cube, data = cube_and_raw('advs.fits')

    s,y,x = (cube._pix_size_slice(ii) for ii in range(3))

    assert s>0
    assert y>0
    assert x>0

    cube.wcs.wcs.cdelt *= -1
    s,y,x = (cube._pix_size_slice(ii) for ii in range(3))

    assert s>0
    assert y>0
    assert x>0

    cube.wcs.wcs.pc *= -1
    s,y,x = (cube._pix_size_slice(ii) for ii in range(3))

    assert s>0
    assert y>0
    assert x>0


def test_varyres_moment_logic_issue364():
    """ regression test for issue364 """
    cube, data = cube_and_raw('vda_beams.fits')

    assert isinstance(cube, VaryingResolutionSpectralCube)

    # the beams are very different, but for this test we don't care
    cube.beam_threshold = 1.0

    with warnings.catch_warnings(record=True) as wrn:
        warnings.simplefilter('default')
        # note that cube.moment(order=0) is different from cube.moment0()
        # because cube.moment0() calls cube.moment(order=0, axis=(whatever)),
        # but cube.moment doesn't necessarily have to receive the axis kwarg
        m0 = cube.moment(order=0)

    if six.PY2:
        # sad face, tests do not work
        pass
    else:
        assert "Arithmetic beam averaging is being performed" in str(wrn[-1].message)
    assert_quantity_allclose(m0.meta['beam'].major, 0.25*u.arcsec)


def test_mask_bad_beams():
    cube, data = cube_and_raw('vda_beams.fits')

    # make sure all of the beams are initially good (finite)
    assert np.all(cube.goodbeams_mask)
    # make sure cropping the cube maintains the mask
    assert np.all(cube[:3].goodbeams_mask)

    # middle two beams have same area
    masked_cube = cube.mask_out_bad_beams(0.01,
                                          reference_beam=Beam(0.3*u.arcsec,
                                                              0.2*u.arcsec,
                                                              60*u.deg))

    assert np.all(masked_cube.mask.include()[:,0,0] == [False,False,True,False])
    assert np.all(masked_cube.goodbeams_mask == [False,False,True,False])

    mean = masked_cube.mean(axis=0)
    assert np.all(mean == cube[2,:,:])


    masked_cube2 = cube.mask_out_bad_beams(0.5,)

    mean2 = masked_cube2.mean(axis=0)
    assert np.all(mean2 == (cube[2,:,:]+cube[1,:,:])/2)
    assert np.all(masked_cube2.goodbeams_mask == [False,True,True,False])


def test_convolve_to():
    cube, data = cube_and_raw('vda_beams.fits')

    convolved = cube.convolve_to(Beam(0.5*u.arcsec))


def test_convolve_to_with_bad_beams():
    cube, data = cube_and_raw('vda_beams.fits')

    convolved = cube.convolve_to(Beam(0.5*u.arcsec))


    with pytest.raises(ValueError) as exc:
        # should not work: biggest beam is 0.4"
        convolved = cube.convolve_to(Beam(0.35*u.arcsec))

    assert exc.value.args[0] == "Beam could not be deconvolved"


    # middle two beams are smaller than 0.4
    masked_cube = cube.mask_channels([False, True, True, False])

    # should work: biggest beam is 0.3 arcsec (major)
    convolved = masked_cube.convolve_to(Beam(0.35*u.arcsec))

    # this is a copout test; should really check for correctness...
    assert np.all(np.isfinite(convolved.filled_data[1:3]))

def test_jybeam_factors():
    cube, data = cube_and_raw('vda_beams.fits')

    assert_allclose(cube.jtok_factors(),
                    [15111171.12641629, 10074201.06746361, 10074287.73828087,
                     15111561.14508185])

def test_channelmask_singlebeam():
    cube, data = cube_and_raw('adv.fits')

    masked_cube = cube.mask_channels([False, True, True, False])

    assert np.all(masked_cube.mask.include()[:,0,0] == [False, True, True, False])

def test_mad_std():
    cube, data = cube_and_raw('adv.fits')

    if int(astropy.__version__[0]) < 2:
        with pytest.raises(NotImplementedError) as exc:
            cube.mad_std()

    else:
        # mad_std run manually on data
        result = np.array([[0.15509701,  0.45763670],
                           [0.55907956,  0.42932451],
                           [0.48819454,  0.25499305]])

        np.testing.assert_almost_equal(cube.mad_std(axis=0).value, result)

        mcube = cube.with_mask(cube < 0.98*u.K)

        result2 = np.array([[0.15509701,  0.45763670],
                            [0.55907956,  0.23835865],
                            [0.48819454,  0.25499305]])

        np.testing.assert_almost_equal(mcube.mad_std(axis=0).value, result2)

def test_mad_std_params():
    cube, data = cube_and_raw('adv.fits')

    # mad_std run manually on data
    result = np.array([[0.15509701,  0.45763670],
                       [0.55907956,  0.42932451],
                       [0.48819454,  0.25499305]])

    np.testing.assert_almost_equal(cube.mad_std(axis=0, how='cube').value, result)
    np.testing.assert_almost_equal(cube.mad_std(axis=0, how='ray').value, result)

    with pytest.raises(NotImplementedError) as exc:
        cube.mad_std(axis=0, how='slice')

    with pytest.raises(NotImplementedError) as exc:
        cube.mad_std(axis=1, how='slice')

    with pytest.raises(NotImplementedError) as exc:
        cube.mad_std(axis=(1,2), how='ray')

    # stats.mad_std(data, axis=(1,2))
    np.testing.assert_almost_equal(cube.mad_std(axis=0, how='ray').value, result)


def test_caching():

    cube, data = cube_and_raw('adv.fits')

    assert len(cube._cache) == 0

    worldextrema = cube.world_extrema

    assert len(cube._cache) == 1

    # see https://stackoverflow.com/questions/46181936/access-a-parent-class-property-getter-from-the-child-class
    world_extrema_function = base_class.SpatialCoordMixinClass.world_extrema.fget.wrapped_function

    assert cube.world_extrema is cube._cache[(world_extrema_function, ())]
    np.testing.assert_almost_equal(worldextrema.value,
                                   cube.world_extrema.value)

def test_spatial_smooth_g2d():
    cube, data = cube_and_raw('adv.fits')

    #
    # Guassian 2D smoothing test
    #
    g2d = Gaussian2DKernel(3)
    cube_g2d = cube.spatial_smooth(g2d)

    # Check first slice
    result0 = np.array([[ 0.06653894,  0.06598313],
                        [ 0.07206352,  0.07151016],
                        [ 0.0702898 ,  0.0697944 ]])

    np.testing.assert_almost_equal(cube_g2d[0].value, result0)

    # Check third slice
    result2 = np.array([[ 0.04217102,  0.04183251],
                        [ 0.04470876,  0.04438826],
                        [ 0.04269588,  0.04242956]])

    np.testing.assert_almost_equal(cube_g2d[2].value, result2)

def test_spatial_smooth_preserves_unit():
    """
    Regression test for issue527
    """
    cube, data = cube_and_raw('adv.fits')
    cube._unit = u.K

    #
    # Guassian 2D smoothing test
    #
    g2d = Gaussian2DKernel(3)
    cube_g2d = cube.spatial_smooth(g2d)

    assert cube_g2d.unit == u.K

def test_spatial_smooth_t2d():
    cube, data = cube_and_raw('adv.fits')

    #
    # Tophat 2D smoothing test
    #
    t2d = Tophat2DKernel(3)
    cube_t2d = cube.spatial_smooth(t2d)

    # Check first slice
    result0 = np.array([[ 0.14864167,  0.14864167],
                        [ 0.14864167,  0.14864167],
                        [ 0.14864167,  0.14864167]])

    np.testing.assert_almost_equal(cube_t2d[0].value, result0)

    # Check third slice
    result2 = np.array([[ 0.09203958,  0.09203958],
                        [ 0.09203958,  0.09203958],
                        [ 0.09203958,  0.09203958]])

    np.testing.assert_almost_equal(cube_t2d[2].value, result2)


def test_spatial_smooth_median():

    pytest.importorskip('scipy.ndimage')

    cube, data = cube_and_raw('adv.fits')

    cube_median = cube.spatial_smooth_median(3)

    # Check first slice
    result0 = np.array([[ 0.54671028,  0.54671028],
                        [ 0.89482735,  0.77513282],
                        [ 0.93949894,  0.89482735]])

    np.testing.assert_almost_equal(cube_median[0].value, result0)

    # Check third slice
    result2 = np.array([[ 0.38867729,  0.35675333],
                        [ 0.38867729,  0.35675333],
                        [ 0.35675333,  0.54269608]])

    np.testing.assert_almost_equal(cube_median[2].value, result2)


def test_spectral_smooth_median():

    pytest.importorskip('scipy.ndimage')

    cube, data = cube_and_raw('adv.fits')

    cube_spectral_median = cube.spectral_smooth_median(3)

    # Check first slice
    result = np.array([0.77513282,  0.35675333,  0.35675333,  0.98688694])

    np.testing.assert_almost_equal(cube_spectral_median[:,1,1].value, result)


def test_spectral_smooth_median_4cores():

    pytest.importorskip('joblib')
    pytest.importorskip('scipy.ndimage')

    cube, data = cube_and_raw('adv.fits')

    cube_spectral_median = cube.spectral_smooth_median(3, num_cores=4)

    # Check first slice
    result = np.array([0.77513282,  0.35675333,  0.35675333,  0.98688694])

    np.testing.assert_almost_equal(cube_spectral_median[:,1,1].value, result)

def test_initialization_from_units():
    """
    Regression test for issue 447
    """
    cube, data = cube_and_raw('adv.fits')

    newcube = SpectralCube(data=cube.filled_data[:], wcs=cube.wcs)

    assert newcube.unit == cube.unit

def test_varyres_spectra():
    cube, data = cube_and_raw('vda_beams.fits')

    assert isinstance(cube, VaryingResolutionSpectralCube)

    sp = cube[:,0,0]

    assert isinstance(sp, VaryingResolutionOneDSpectrum)
    assert hasattr(sp, 'beams')

    sp = cube.mean(axis=(1,2))

    assert isinstance(sp, VaryingResolutionOneDSpectrum)
    assert hasattr(sp, 'beams')


def test_median_2axis():
    """
    As of this writing the bottleneck.nanmedian did not accept an axis that is a
    tuple/list so this test is to make sure that is properly taken into account.
    """
    cube, data = cube_and_raw('adv.fits')

    cube_median = cube.median(axis=(1, 2))

    # Check first slice
    result0 = np.array([0.83498009, 0.2606566 , 0.37271531, 0.48548023])

    np.testing.assert_almost_equal(cube_median.value, result0)


def test_varyres_mask():
    cube, data = cube_and_raw('vda_beams.fits')

    # mask out two beams
    goodbeams = cube.identify_bad_beams(0.5)
    assert all(goodbeams == np.array([False, True, True, False]))

    mcube = cube.mask_out_bad_beams(0.5)
    assert hasattr(mcube, '_goodbeams_mask')
    assert all(mcube.goodbeams_mask == goodbeams)
    assert len(mcube.beams) == 2

    sp_masked = mcube[:,0,0]

    assert hasattr(sp_masked, '_goodbeams_mask')
    assert all(sp_masked.goodbeams_mask == goodbeams)
    assert len(sp_masked.beams) == 2

    try:
        assert mcube.unmasked_beams == cube.beams
    except ValueError:
        # older versions of beams
        assert np.all(mcube.unmasked_beams == cube.beams)

    try:
        # check that slicing works too
        assert mcube[:5].unmasked_beams == cube[:5].beams
    except ValueError:
        assert np.all(mcube[:5].unmasked_beams == cube[:5].beams)
