"""
Performance-related tests to make sure we don't use more memory than we should
"""

from __future__ import print_function, absolute_import, division

import numpy as np

import pytest
import tempfile
import sys

try:
    import tracemalloc
    tracemallocOK = True
except ImportError:
    tracemallocOK = False

# The comparison of Quantities in test_memory_usage
# fail with older versions of numpy
from distutils.version import LooseVersion

NPY_VERSION_CHECK = LooseVersion(np.version.version) >= "1.13"

from .test_moments import moment_cube
from .helpers import assert_allclose
from ..spectral_cube import SpectralCube
from . import utilities

from astropy import convolution, units as u

import itertools

@pytest.mark.parametrize('use_dask, use_memmap, num_cores, parallel, verbose'.split(', '),
                         itertools.product([True,False], repeat=5))
def test_parallel_smoothing_spatial(use_dask, use_memmap, num_cores, parallel, verbose):

    cube,_ = utilities.generate_gaussian_cube(shape=(4,32,32))

    basic_result = cube.spatial_smooth(kernel=convolution.Gaussian2DKernel(2.0),
                                       use_dask=False, use_memmap=False,
                                       num_cores=1, parallel=False)

    result = cube.spatial_smooth(kernel=convolution.Gaussian2DKernel(2.0),
                                 use_dask=use_dask, use_memmap=use_memmap,
                                 num_cores=num_cores, parallel=parallel,
                                 verbose=verbose)

    np.testing.assert_array_almost_equal(basic_result.unitless_filled_data[:],
                                         result.unitless_filled_data[:])

def test_dask_apply_to_images():
    cube,_ = utilities.generate_gaussian_cube(shape=(4,32,32))

    basic_result = cube.spatial_smooth(kernel=convolution.Gaussian2DKernel(2.0),
                                       use_dask=False, use_memmap=False,
                                       num_cores=1, parallel=False)


    result_nomemmap = cube.dask_apply_function_by_image(function=convolution.convolve,
                                                        kernel=convolution.Gaussian2DKernel(2.0),
                                                        projection=False,
                                                        reduce=False,
                                                        use_memmap=False)

    np.testing.assert_array_almost_equal(basic_result.unitless_filled_data[:],
                                         result_nomemmap.unitless_filled_data[:])

    result = cube.dask_apply_function_by_image(function=convolution.convolve,
                                               kernel=convolution.Gaussian2DKernel(2.0),
                                               projection=False, reduce=False)

    np.testing.assert_array_almost_equal(basic_result.unitless_filled_data[:],
                                         result.unitless_filled_data[:])
