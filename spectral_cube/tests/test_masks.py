import numpy as np
from numpy.testing import assert_allclose

from ..spectral_cube import SpectralCubeMask, FunctionMask

def test_spectral_cube_mask():

    mask = np.array([[[False, True, True, False, True]]])
    mask_wcs = None

    m = SpectralCubeMask(mask, mask_wcs)

    data = np.arange(5).reshape((1,1,5))
    wcs = None

    assert_allclose(m.include(data, wcs), [[[0,1,1,0,1]]])
    assert_allclose(m.exclude(data, wcs), [[[1,0,0,1,0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan,1,2,np.nan,4]]])

    assert_allclose(m.include(data, wcs, slices=(0,0,slice(1,4))), [1,1,0])
    assert_allclose(m.exclude(data, wcs, slices=(0,0,slice(1,4))), [0,0,1])
    assert_allclose(m._filled(data, wcs, slices=(0,0,slice(1,4))), [1,2,np.nan])