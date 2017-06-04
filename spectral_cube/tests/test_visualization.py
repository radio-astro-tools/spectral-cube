from __future__ import print_function, absolute_import, division

import pytest

try:
    import pvextractor
    PVEXTRACTOR_INSTALLED = True
except ImportError:
    PVEXTRACTOR_INSTALLED = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

try:
    import aplpy
    APLPY_INSTALLED = True
except ImportError:
    APLPY_INSTALLED = False

from .. import (SpectralCube, BooleanArrayMask, FunctionMask, LazyMask,
                CompositeMask)
from ..spectral_cube import OneDSpectrum, Projection
from ..np_compat import allbadtonan
from .. import spectral_axis

from .test_spectral_cube import cube_and_raw


@pytest.mark.skipif("not PVEXTRACTOR_INSTALLED")
def test_to_pvextractor():

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    pv = cube.to_pvextractor()


@pytest.mark.skipif("not MATPLOTLIB_INSTALLED")
def test_projvis_noaplpy():

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    mom0 = cube.moment0()
    mom0.quicklook(use_aplpy=False)

@pytest.mark.skipif("not APLPY_INSTALLED")
def test_projvis():

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    mom0 = cube.moment0()
    mom0.quicklook(use_aplpy=True)

@pytest.mark.skipif("not APLPY_INSTALLED")
def test_mask_quicklook():

    cube, data = cube_and_raw('vda_Jybeam_lower.fits')

    cube.mask.quicklook(view=(0, slice(None), slice(None)), use_aplpy=True)
