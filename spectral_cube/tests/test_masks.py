import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS

from ..spectral_cube import SpectralCubeMask, FunctionMask

def test_spectral_cube_mask():

    mask = np.array([[[False, True, True, False, True]]])
    mask_wcs = WCS()

    m = SpectralCubeMask(mask, mask_wcs)

    data = np.arange(5).reshape((1,1,5))
    wcs = WCS()

    assert_allclose(m.include(data, wcs), [[[0,1,1,0,1]]])
    assert_allclose(m.exclude(data, wcs), [[[1,0,0,1,0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan,1,2,np.nan,4]]])
    assert_allclose(m._flattened(data, wcs), [1,2,4])

    assert_allclose(m.include(data, wcs, slices=(0,0,slice(1,4))), [1,1,0])
    assert_allclose(m.exclude(data, wcs, slices=(0,0,slice(1,4))), [0,0,1])
    assert_allclose(m._filled(data, wcs, slices=(0,0,slice(1,4))), [1,2,np.nan])
    assert_allclose(m._flattened(data, wcs, slices=(0,0,slice(1,4))), [1,2])


def test_function_mask():

    m = FunctionMask(lambda x: x > 2)

    data = np.arange(5).reshape((1,1,5))
    wcs = WCS()

    assert_allclose(m.include(data, wcs), [[[0,0,0,1,1]]])
    assert_allclose(m.exclude(data, wcs), [[[1,1,1,0,0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan,np.nan,np.nan,3,4]]])
    assert_allclose(m._flattened(data, wcs), [3,4])

    assert_allclose(m.include(data, wcs, slices=(0,0,slice(1,4))), [0,0,1])
    assert_allclose(m.exclude(data, wcs, slices=(0,0,slice(1,4))), [1,1,0])
    assert_allclose(m._filled(data, wcs, slices=(0,0,slice(1,4))), [np.nan,np.nan,3])
    assert_allclose(m._flattened(data, wcs, slices=(0,0,slice(1,4))), [3])
