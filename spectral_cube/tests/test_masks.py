from __future__ import print_function, absolute_import, division

import pytest
import itertools
import operator
import numpy as np
from numpy.testing import assert_allclose
from numpy.lib.stride_tricks import as_strided
from astropy.wcs import WCS
from astropy import units as u

from .test_spectral_cube import cube_and_raw
from .. import (BooleanArrayMask, LazyMask, LazyComparisonMask,
                FunctionMask, CompositeMask)
from ..masks import is_broadcastable_and_smaller, dims_to_skip, view_of_subset

from distutils.version import LooseVersion


def test_spectral_cube_mask():

    mask = np.array([[[False, True, True, False, True]]])
    mask_wcs = WCS()

    m = BooleanArrayMask(mask, mask_wcs)

    data = np.arange(5).reshape((1, 1, 5))
    wcs = WCS()

    assert_allclose(m.include(data, wcs), [[[0, 1, 1, 0, 1]]])
    assert_allclose(m.exclude(data, wcs), [[[1, 0, 0, 1, 0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan, 1, 2, np.nan, 4]]])
    assert_allclose(m._flattened(data, wcs), [1, 2, 4])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [1, 2, np.nan])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [1, 2])


def test_lazy_mask():

    data = np.arange(5).reshape((1, 1, 5))
    wcs = WCS()

    m = LazyMask(lambda x: x > 2, data=data, wcs=wcs)

    assert_allclose(m.include(data, wcs), [[[0, 0, 0, 1, 1]]])
    assert_allclose(m.exclude(data, wcs), [[[1, 1, 1, 0, 0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan, np.nan, np.nan, 3, 4]]])
    assert_allclose(m._flattened(data, wcs), [3, 4])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [np.nan, np.nan, 3])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [3])

    # Now if we call with different data, the results for include and exclude
    # should *not* change.

    data = (3 - np.arange(5)).reshape((1, 1, 5))

    assert_allclose(m.include(data, wcs), [[[0, 0, 0, 1, 1]]])
    assert_allclose(m.exclude(data, wcs), [[[1, 1, 1, 0, 0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan, np.nan, np.nan, 0, -1]]])
    assert_allclose(m._flattened(data, wcs), [0, -1])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [np.nan, np.nan, 0])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [0])


def test_lazy_comparison_mask():

    data = np.arange(5).reshape((1, 1, 5))
    wcs = WCS()

    m = LazyComparisonMask(operator.gt, 2, data=data, wcs=wcs)

    assert_allclose(m.include(data, wcs), [[[0, 0, 0, 1, 1]]])
    assert_allclose(m.exclude(data, wcs), [[[1, 1, 1, 0, 0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan, np.nan, np.nan, 3, 4]]])
    assert_allclose(m._flattened(data, wcs), [3, 4])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [np.nan, np.nan, 3])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [3])

    # Now if we call with different data, the results for include and exclude
    # should *not* change.

    data = (3 - np.arange(5)).reshape((1, 1, 5))

    assert_allclose(m.include(data, wcs), [[[0, 0, 0, 1, 1]]])
    assert_allclose(m.exclude(data, wcs), [[[1, 1, 1, 0, 0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan, np.nan, np.nan, 0, -1]]])
    assert_allclose(m._flattened(data, wcs), [0, -1])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [np.nan, np.nan, 0])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [0])


def test_function_mask_incorrect_shape():

    # The following function will return the incorrect shape because it does
    # not apply the view
    def threshold(data, wcs, view=()):
        return data > 2

    m = FunctionMask(threshold)

    data = np.arange(5).reshape((1, 1, 5))
    wcs = WCS()

    with pytest.raises(ValueError) as exc:
        m.include(data, wcs, view=(0, 0, slice(1, 4)))
    assert exc.value.args[0] == "Function did not return mask with correct shape - expected (3,), got (1, 1, 5)"


def test_function_mask():

    def threshold(data, wcs, view=()):
        return data[view] > 2

    m = FunctionMask(threshold)

    data = np.arange(5).reshape((1, 1, 5))
    wcs = WCS()

    assert_allclose(m.include(data, wcs), [[[0, 0, 0, 1, 1]]])
    assert_allclose(m.exclude(data, wcs), [[[1, 1, 1, 0, 0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan, np.nan, np.nan, 3, 4]]])
    assert_allclose(m._flattened(data, wcs), [3, 4])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [np.nan, np.nan, 3])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [3])

    # Now if we call with different data, the results for include and exclude
    # *should* change.

    data = (3 - np.arange(5)).reshape((1, 1, 5))

    assert_allclose(m.include(data, wcs), [[[1, 0, 0, 0, 0]]])
    assert_allclose(m.exclude(data, wcs), [[[0, 1, 1, 1, 1]]])
    assert_allclose(m._filled(data, wcs), [[[3, np.nan, np.nan, np.nan, np.nan]]])
    assert_allclose(m._flattened(data, wcs), [3])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(0, 3))), [1, 0, 0])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(0, 3))), [0, 1, 1])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(0, 3))), [3, np.nan, np.nan])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(0, 3))), [3])


def test_composite_mask():

    def lower_threshold(data, wcs, view=()):
        return data[view] > 0

    def upper_threshold(data, wcs, view=()):
        return data[view] < 3

    m1 = FunctionMask(lower_threshold)
    m2 = FunctionMask(upper_threshold)

    m = m1 & m2

    data = np.arange(5).reshape((1, 1, 5))
    wcs = WCS()

    assert_allclose(m.include(data, wcs), [[[0, 1, 1, 0, 0]]])
    assert_allclose(m.exclude(data, wcs), [[[1, 0, 0, 1, 1]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan, 1, 2, np.nan, np.nan]]])
    assert_allclose(m._flattened(data, wcs), [1, 2])

    assert_allclose(m.include(data, wcs, view=(0, 0, slice(1, 4))), [1, 1, 0])
    assert_allclose(m.exclude(data, wcs, view=(0, 0, slice(1, 4))), [0, 0, 1])
    assert_allclose(m._filled(data, wcs, view=(0, 0, slice(1, 4))), [1, 2, np.nan])
    assert_allclose(m._flattened(data, wcs, view=(0, 0, slice(1, 4))), [1, 2])


def test_mask_logic():

    data = np.arange(5).reshape((1, 1, 5))
    wcs = WCS()

    def threshold_1(data, wcs, view=()):
        return data[view] > 0

    def threshold_2(data, wcs, view=()):
        return data[view] < 4

    def threshold_3(data, wcs, view=()):
        return data[view] != 2

    m1 = FunctionMask(threshold_1)
    m2 = FunctionMask(threshold_2)
    m3 = FunctionMask(threshold_3)

    m = m1 & m2
    assert_allclose(m.include(data, wcs), [[[0, 1, 1, 1, 0]]])

    m = m1 | m2
    assert_allclose(m.include(data, wcs), [[[1, 1, 1, 1, 1]]])

    m = m1 | ~m2
    assert_allclose(m.include(data, wcs), [[[0, 1, 1, 1, 1]]])

    m = m1 & m2 & m3
    assert_allclose(m.include(data, wcs), [[[0, 1, 0, 1, 0]]])

    m = (m1 | m3) & m2
    assert_allclose(m.include(data, wcs), [[[1, 1, 1, 1, 0]]])

    m = m1 ^ m2
    assert_allclose(m.include(data, wcs), [[[1, 0, 0, 0, 1]]])

    m = m1 ^ m3
    assert_allclose(m.include(data, wcs), [[[1, 0, 1, 0, 0]]])


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(('filename'),
                         (('data_advs'),
                          ('data_dvsa'),
                          ('data_sdav'),
                          ('data_sadv'),
                          ('data_vsad'),
                          ('data_vad'),
                          ('data_adv'),
                          ), indirect=['filename'])
def test_mask_spectral_unit(filename, use_dask):
    cube, data = cube_and_raw(filename, use_dask=use_dask)
    mask = BooleanArrayMask(data, cube._wcs)
    mask_freq = mask.with_spectral_unit(u.Hz)

    assert mask_freq._wcs.wcs.ctype[mask_freq._wcs.wcs.spec] == 'FREQ-W2F'

    # values taken from header
    rest = 1.42040571841E+09*u.Hz
    crval = -3.21214698632E+05*u.m/u.s
    outcv = crval.to(u.m, u.doppler_optical(rest)).to(u.Hz, u.spectral())

    assert_allclose(mask_freq._wcs.wcs.crval[mask_freq._wcs.wcs.spec],
                    outcv.to(u.Hz).value)


def test_wcs_validity_check(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)
    mask = BooleanArrayMask(data > 0, cube._wcs)
    cube = cube.with_mask(mask)
    s2 = cube.spectral_slab(-2 * u.km / u.s, 2 * u.km / u.s)
    s3 = s2.with_spectral_unit(u.km / u.s, velocity_convention=u.doppler_radio)
    # just checking that this works, not that it does anything in particular
    moment_map = s3.moment(order=1)


def test_wcs_validity_check_failure(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    assert cube.wcs.wcs.crval[2] == -3.21214698632E+05

    wcs2 = cube.wcs.deepcopy()
    # add some difference in the 5th decimal place
    wcs2.wcs.crval[2] += 0.00001

    assert wcs2.wcs.crval[2] == -3.21214698622E+05
    assert cube.wcs.wcs.crval[2] == -3.21214698632E+05

    # can make a mask
    mask = BooleanArrayMask(data>0, wcs2)

    assert cube.wcs.wcs.crval[2] != wcs2.wcs.crval[2]
    assert cube._wcs.wcs.crval[2] != wcs2.wcs.crval[2]

    # but if it's not exactly equal, an error should be raised at this step
    with pytest.raises(ValueError, match="WCS does not match mask WCS"):
        cube.with_mask(mask)

    # this one should work though
    cube = cube.with_mask(mask, wcs_tolerance=1e-4)
    assert cube._wcs_tolerance == 1e-4

    # then the rest of this should be OK
    s2 = cube.spectral_slab(-2 * u.km / u.s, 2 * u.km / u.s)
    s3 = s2.with_spectral_unit(u.km / u.s, velocity_convention=u.doppler_radio)
    # just checking that this works, not that it does anything in particular
    moment_map = s3.moment(order=1)


def test_mask_spectral_unit_functions(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    # function mask should do nothing
    mask1 = FunctionMask(lambda x: x>0)
    mask_freq1 = mask1.with_spectral_unit(u.Hz)

    # lazy mask behaves like booleanarraymask
    mask2 = LazyMask(lambda x: x>0, cube=cube)
    mask_freq2 = mask2.with_spectral_unit(u.Hz)

    assert mask_freq2._wcs.wcs.ctype[mask_freq2._wcs.wcs.spec] == 'FREQ-W2F'

    # values taken from header
    rest = 1.42040571841E+09*u.Hz
    crval = -3.21214698632E+05*u.m/u.s
    outcv = crval.to(u.m, u.doppler_optical(rest)).to(u.Hz, u.spectral())

    assert_allclose(mask_freq2._wcs.wcs.crval[mask_freq2._wcs.wcs.spec],
                    outcv.to(u.Hz).value)

    # again, test that it works
    mask3 = CompositeMask(mask1,mask2)
    mask_freq3 = mask3.with_spectral_unit(u.Hz)

    mask_freq3 = CompositeMask(mask_freq1,mask_freq2)
    mask_freq_freq3 = mask_freq3.with_spectral_unit(u.Hz)

    # this one should fail
    #failedmask = CompositeMask(mask_freq1,mask2)

def is_broadcastable_try(shp1, shp2):
    """
    Test whether an array shape can be broadcast to another
    (this is the try/fail approach, which is guaranteed right.... right?)
    http://stackoverflow.com/questions/24743753/test-if-an-array-is-broadcastable-to-a-shape/24745359#24745359
    """
    #This variant does not work as of np 1.10: the strided arrays aren't
    #writable and therefore apparently cannot be broadcast
    # x = np.array([1])
    # a = as_strided(x, shape=shp1, strides=[0] * len(shp1))
    # b = as_strided(x, shape=shp2, strides=[0] * len(shp2))
    a = np.ones(shp1)
    b = np.ones(shp2)
    try:
        c = np.broadcast_arrays(a, b)
        # reverse order: compare last dim first (as broadcasting does)
        if any(bi<ai for ai,bi in zip(shp1,shp2)):
            # don't allow "b" to be smaller than "a"
            return False
        return True
    except ValueError:
        return False

shapes = ([1,5,5], [1,5,1], [5,5,1], [5,5], [5,5,2],
          [2,3,4], [4,3,2], [4,2,3], [2,4,3])
shape_combos = list(itertools.combinations(shapes,2))


@pytest.mark.parametrize(('shp1','shp2'),shape_combos)
def test_is_broadcastable(shp1, shp2):
    assert is_broadcastable_and_smaller(shp1,shp2) == is_broadcastable_try(shp1,shp2)


@pytest.mark.parametrize(('shp1','shp2','dim'),
                         (([5,5],[2,5,5],[0]),
                          #([5,5],[5,5,2],[2]),
                          ([2,5,5],[2,5,5],[])))
def test_dims_to_skip(shp1, shp2, dim):
    assert dims_to_skip(shp1, shp2) == dim


@pytest.mark.parametrize(('shp1','shp2', 'inview', 'outview'),
                         (([5,5],[2,5,5],  (slice(0,1), slice(1,3), slice(2,4),), (slice(1,3), slice(2,4))),
                          # not a valid broadcast ([5,5],[5,5,2],  [slice(1,3), slice(2,4), slice(0,1),], [slice(1,3), slice(2,4)]),
                          ([2,5,5],[2,5,5],(slice(0,1), slice(1,3), slice(2,4),), (slice(0,1), slice(1,3), slice(2,4),))))
def test_view_of_subset(shp1, shp2, inview, outview):
    assert view_of_subset(shp1,shp2,inview) == outview


def test_flat_mask(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    mask_array = np.array([[True,False],[False,False],[True,True]])
    bool_mask_array = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                       shape=cube.shape)
    mcube = cube.with_mask(bool_mask_array)

    # I think we can use == instead of 'almost equal' here because the
    # underlying data should be identical (the same in memory)

    assert np.all(cube.sum(axis=0)[mask_array] == mcube.sum(axis=0)[mask_array])
    assert np.all(np.isnan(mcube.sum(axis=0)[~mask_array]))


@pytest.mark.skipif(LooseVersion(np.__version__) < LooseVersion('1.7'),
                    reason='Numpy <1.7 does not support multi-slice indexing.')
def test_flat_mask_spectral(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    mask_array = np.array([[True,False],[False,False],[True,True]])
    bool_mask_array = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                       shape=cube.shape)
    mcube = cube.with_mask(bool_mask_array)

    # Broadcast the 2D mask to 3D
    cubemask = (np.ones(4,dtype='bool')[:,None,None] & mask_array[None,:,:])
    # Check that spectral masking works too
    assert np.all((data*cubemask).sum(axis=(1,2)) ==
                  mcube.sum(axis=(1,2)).value)


def test_include(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    mask_array = np.array([[True,False],[False,False],[True,True]])
    bool_mask_array = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                       shape=cube.shape)

    assert np.all(bool_mask_array.include() == mask_array)


def test_1d_mask(data_adv, use_dask):
    # regression test for issue revealed in #183
    # In principle, this is also a regression test for #298, except this always
    # passed where #298 failed, which I don't understand.

    cube, data = cube_and_raw(data_adv, use_dask=use_dask)
    mask = np.array([True, False, True, False])

    sum0 = cube.with_mask(mask[:,None,None]).sum(axis=0)
    sum0d = data[mask, :,:].sum(axis=0)

    np.testing.assert_almost_equal(sum0.value, sum0d)


def test_1d_mask_amp(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)
    mask = np.array([True, False, True, False])

    Mask = BooleanArrayMask(mask[:,None,None],
                            wcs=cube._wcs,
                            shape=cube.shape,
                           )

    ampd = cube.mask & Mask

    ampd.include()


def test_2dcomparison_mask_1d_index(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    med = cube.median(axis=0)
    mask = cube > med

    mcube = cube.with_mask(mask)

    assert all(mask[:,1,1].include() ==
               mask.include()[:,1,1])

    spec = mcube[:,1,1]

    assert spec.ndim == 1

    assert all(spec.mask.include() == mask.include()[:,1,1])

    assert spec[:-1].mask.include().shape == (3,)
    assert all(spec[:-1].mask.include() == mask.include()[:-1,1,1])

    assert isinstance(spec[0], u.Quantity)

    spec = mcube[:-1,1,1]

    assert spec.ndim == 1
    assert hasattr(spec, '_fill_value')

    assert all(spec.mask.include() == mask.include()[:-1,1,1])

    assert spec[:-1].mask.include().shape == (2,)
    assert all(spec[:-1].mask.include() == mask.include()[:-2,1,1])

    assert isinstance(spec[0], u.Quantity)


def test_1dcomparison_mask_1d_index(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    med = cube.median()
    mask = cube > med

    mcube = cube.with_mask(mask)

    assert all(mask[:,1,1].include() ==
               mask.include()[:,1,1])

    spec = mcube[:,1,1]

    assert spec.ndim == 1

    assert all(spec.mask.include() == [True,False,False,True])

    assert spec[:-1].mask.include().shape == (3,)
    assert all(spec[:-1].mask.include() == [True,False,False])

    assert isinstance(spec[0], u.Quantity)


def test_1dmask_indexing(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    med = cube.median()
    mask = cube > med

    mcube = cube.with_mask(mask)

    assert all(mask[:,1,1].include() ==
               mask.include()[:,1,1])

    spec = mcube[:,1,1]

    badvals = np.array([False,True,True,False], dtype='bool')
    assert np.all(np.isnan(spec[badvals]))
    assert not np.any(np.isnan(spec[~badvals]))


def test_numpy_ma_tools(data_adv, use_dask):
    """
    check that np.ma.core.is_masked works
    """

    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    med = cube.median()
    mask = cube > med

    mcube = cube.with_mask(mask)

    assert np.ma.core.is_masked(mcube)
    assert np.ma.core.getmask(mcube) is not None
    assert np.ma.core.is_masked(mcube[:,0,0])
    assert np.ma.core.getmask(mcube[:,0,0]) is not None


@pytest.mark.xfail
def test_numpy_ma_tools_2d(data_adv, use_dask):
    """ This depends on 2D objects keeping masks, which depends on #395.
    so, TODO: un-xfail this """

    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    med = cube.median()
    mask = cube > med

    mcube = cube.with_mask(mask)

    assert np.ma.core.is_masked(mcube[0,:,:])
    assert np.ma.core.getmask(mcube[0,:,:]) is not None


def test_filled(data_adv, use_dask):
    """ test that 'filled' works """

    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    med = cube.median()
    mask = cube > med

    mcube = cube.with_mask(mask)
    assert np.isnan(mcube._fill_value)

    filled = mcube.filled(np.nan)
    filled_ = mcube.filled()
    assert_allclose(filled, filled_)

    assert (np.isnan(filled) == mcube.mask.exclude()).all()


def test_boolean_array_composite_mask(data_adv, use_dask):

    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    med = cube.median()
    mask = cube > med
    arrmask = cube.max(axis=0) > med

    # we're just testing that this doesn't fail
    combined_mask = mask & arrmask

    mcube = cube.with_mask(combined_mask)

    # not doing assert_almost_equal because I don't want to worry about precision
    assert (mcube.sum() > 9.0 * u.K) & (mcube.sum() < 9.1*u.K)
