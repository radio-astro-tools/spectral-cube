import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS

from ..spectral_cube import SpectralCubeMask, LazyMask, FunctionMask


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


def test_lazy_mask():

    data = np.arange(5).reshape((1,1,5))
    wcs = WCS()

    m = LazyMask(lambda x: x > 2, data, wcs)

    assert_allclose(m.include(data, wcs), [[[0,0,0,1,1]]])
    assert_allclose(m.exclude(data, wcs), [[[1,1,1,0,0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan,np.nan,np.nan,3,4]]])
    assert_allclose(m._flattened(data, wcs), [3,4])

    assert_allclose(m.include(data, wcs, slices=(0,0,slice(1,4))), [0,0,1])
    assert_allclose(m.exclude(data, wcs, slices=(0,0,slice(1,4))), [1,1,0])
    assert_allclose(m._filled(data, wcs, slices=(0,0,slice(1,4))), [np.nan,np.nan,3])
    assert_allclose(m._flattened(data, wcs, slices=(0,0,slice(1,4))), [3])

    # Now if we call with different data, the results for include and exclude
    # should *not* change.

    data = (3-np.arange(5)).reshape((1,1,5))

    assert_allclose(m.include(data, wcs), [[[0,0,0,1,1]]])
    assert_allclose(m.exclude(data, wcs), [[[1,1,1,0,0]]])
    assert_allclose(m._filled(data, wcs), [[[np.nan,np.nan,np.nan,0,-1]]])
    assert_allclose(m._flattened(data, wcs), [0,-1])

    assert_allclose(m.include(data, wcs, slices=(0,0,slice(1,4))), [0,0,1])
    assert_allclose(m.exclude(data, wcs, slices=(0,0,slice(1,4))), [1,1,0])
    assert_allclose(m._filled(data, wcs, slices=(0,0,slice(1,4))), [np.nan,np.nan,0])
    assert_allclose(m._flattened(data, wcs, slices=(0,0,slice(1,4))), [0])


def test_function_mask_incorrect_shape():

    # The following function will return the incorrect shape because it does
    # not treat slices correctly - using slices=None will *add* a dimension.
    def threshold(data, wcs, slices=None):
        return data[slices] > 2

    m = FunctionMask(threshold)

    data = np.arange(5).reshape((1,1,5))
    wcs = WCS()

    with pytest.raises(ValueError) as exc:
        m.include(data, wcs)
    assert exc.value.args[0] == "Function did not return mask with correct shape - expected (1, 1, 5), got (1, 1, 1, 5)"


def test_function_mask():

    def threshold(data, wcs, slices=None):
        if slices is None:
            return data > 2
        else:
            return data[slices] > 2

    m = FunctionMask(threshold)

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

    # Now if we call with different data, the results for include and exclude
    # *should* change.

    data = (3-np.arange(5)).reshape((1,1,5))

    assert_allclose(m.include(data, wcs), [[[1,0,0,0,0]]])
    assert_allclose(m.exclude(data, wcs), [[[0,1,1,1,1]]])
    assert_allclose(m._filled(data, wcs), [[[3, np.nan,np.nan,np.nan,np.nan]]])
    assert_allclose(m._flattened(data, wcs), [3])

    assert_allclose(m.include(data, wcs, slices=(0,0,slice(0,3))), [1,0,0])
    assert_allclose(m.exclude(data, wcs, slices=(0,0,slice(0,3))), [0,1,1])
    assert_allclose(m._filled(data, wcs, slices=(0,0,slice(0,3))), [3,np.nan,np.nan])
    assert_allclose(m._flattened(data, wcs, slices=(0,0,slice(0,3))), [3])
