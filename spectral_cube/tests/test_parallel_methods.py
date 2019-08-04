"""
Parallel function tests for correctness (surprisingly difficult to achieve)
"""

from __future__ import print_function, absolute_import, division

import numpy as np

import pytest

from .helpers import assert_allclose
from . import utilities

from astropy import convolution

import itertools
import pickle

try:
    import dask
    DASK_OK = True
except ImportError:
    DASK_OK = False

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False



# optional fixture to add; client should be closed internally, though.
# @pytest.mark.skipif("not DASK_OK")
# @pytest.fixture(scope="module")
# def dask_client_spinup():
#     import dask.distributed
#     try:
#         client = dask.distributed.get_client()
#     except ValueError:
#         client = dask.distributed.Client()
#     yield client  # provide the fixture value
#     print("teardown dask client")
#     client.close()


pars = 'use_dask, use_memmap, num_cores, parallel, verbose, write_to_disk'.split(', ')
@pytest.mark.parametrize(pars,
                         itertools.product([True,False], repeat=len(pars)))
def test_parallel_smoothing_spatial(use_dask, use_memmap, num_cores, parallel,
                                    verbose, write_to_disk):

    if use_dask and not DASK_OK:
        pytest.skip("Dask not installed")
    if not use_dask and not JOBLIB_OK:
        pytest.skip("joblib not installed")

    # BEWARE: if shape[0] == nprocs, you can get spurious passes
    cube,_ = utilities.generate_gaussian_cube(shape=(6,30,32),
                                              write_to_disk=write_to_disk)

    try:
        result = cube.spatial_smooth(kernel=convolution.Gaussian2DKernel(2.0),
                                     use_dask=use_dask, use_memmap=use_memmap,
                                     num_cores=4 if num_cores else 1, parallel=parallel,
                                     verbose=verbose)
    except ValueError as ex:
        if num_cores == 0 and parallel and not use_dask:
            assert "n_jobs == 0 in Parallel has no meaning" in str(ex)
            return
        elif num_cores > 0 and not parallel:
            assert ("parallel execution was not requested, but multiple cores were: "
                    "these are incompatible options."
                    "  Either specify num_cores=1 or parallel=True") in str(ex)
            return
        else:
            raise ex
    except pickle.PicklingError as ex:
        if "Could not pickle the task to send it to the workers." in str(ex):
            if use_memmap and not use_dask and write_to_disk:
                pytest.xfail("Known error: joblib apparently can't pass around "
                             "memmap objects read from disk.  I don't know how to deal "
                             "with this yet (August 2019)")
                return
        raise

    basic_result = cube.spatial_smooth(kernel=convolution.Gaussian2DKernel(2.0),
                                       use_dask=False, use_memmap=False,
                                       num_cores=1, parallel=False)


    assert_allclose(basic_result.unitless_filled_data[:],
                    result.unitless_filled_data[:])

@pytest.mark.parametrize(pars,
                         itertools.product([True,False], repeat=len(pars)))
def test_parallel_smoothing_spectral(use_dask, use_memmap, num_cores, parallel,
                                     verbose, write_to_disk):

    if use_dask and not DASK_OK:
        pytest.skip("Dask not installed")
    if not use_dask and not JOBLIB_OK:
        pytest.skip("joblib not installed")

    # use asymmetric dimensions to help debugging
    cube,_ = utilities.generate_gaussian_cube(shape=(32,4,6),
                                              write_to_disk=write_to_disk)

    try:
        result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(2.0),
                                      use_dask=use_dask, use_memmap=use_memmap,
                                      num_cores=4 if num_cores else 1, parallel=parallel,
                                      verbose=verbose)
    except ValueError as ex:
        if num_cores == 0 and parallel and not use_dask:
            assert "n_jobs == 0 in Parallel has no meaning" in str(ex)
            return
        elif num_cores > 0 and not parallel:
            assert ("parallel execution was not requested, but multiple cores were: "
                    "these are incompatible options."
                    "  Either specify num_cores=1 or parallel=True") in str(ex)
            return
        else:
            raise ex
    except pickle.PicklingError as ex:
        if "Could not pickle the task to send it to the workers." in str(ex):
            if use_memmap and not use_dask and write_to_disk:
                pytest.xfail("Known error: joblib apparently can't pass around "
                             "memmap objects read from disk.  I don't know how to deal "
                             "with this yet (August 2019)")
                return
        raise

    basic_result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(2.0),
                                        use_dask=False, use_memmap=False,
                                        num_cores=1, parallel=False)


    assert_allclose(basic_result.unitless_filled_data[:],
                    result.unitless_filled_data[:])



# parametrize over the length of the spectra dimension to make sure the mapping
# doesn't go right for the wrong reason
@pytest.mark.skipif('not DASK_OK')
@pytest.mark.parametrize('nspec', (4, 10, 14))
def test_dask_apply_to_images(nspec):
    cube,_ = utilities.generate_gaussian_cube(shape=(nspec,32,32))

    basic_result = cube.spatial_smooth(kernel=convolution.Gaussian2DKernel(2.0),
                                       use_dask=False, use_memmap=False,
                                       num_cores=1, parallel=False)


    result_nomemmap = cube.dask_apply_function_by_image(function=convolution.convolve,
                                                        kernel=convolution.Gaussian2DKernel(2.0),
                                                        projection=False,
                                                        reduce=False,
                                                        use_memmap=False)

    assert_allclose(basic_result.unitless_filled_data[:],
                    result_nomemmap.unitless_filled_data[:])

    result = cube.dask_apply_function_by_image(function=convolution.convolve,
                                               kernel=convolution.Gaussian2DKernel(2.0),
                                               projection=False, reduce=False)

    assert_allclose(basic_result.unitless_filled_data[:],
                    result.unitless_filled_data[:])
