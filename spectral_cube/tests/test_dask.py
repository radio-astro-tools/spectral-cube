# Tests specific to the dask class

import os
from numpy.core.shape_base import block
import pytest
import numpy as np
from unittest.mock import patch

from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as u
from astropy.utils import data

try:
    from distributed.utils_test import client, loop, cluster_fixture, cleanup, loop_in_thread  # noqa
    # zarr & fsspec required for writing to disk w/dask
    import zarr, fsspec  # noqa
    DISTRIBUTED_INSTALLED = True
except ImportError:
    DISTRIBUTED_INSTALLED = False

from spectral_cube import DaskSpectralCube, SpectralCube, DaskVaryingResolutionSpectralCube
from .test_casafuncs import make_casa_testimage

try:
    import casatools
    from casatools import image
    CASA_INSTALLED = True
except ImportError:
    try:
        from taskinit import ia as image
        CASA_INSTALLED = True
    except ImportError:
        CASA_INSTALLED = False

DATA = os.path.join(os.path.dirname(__file__), 'data')


class Array:

    args = None
    kwargs = None

    def compute(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_scheduler(data_adv):

    cube = DaskSpectralCube.read(data_adv)
    fake_array = Array()

    cube._compute(fake_array)
    assert fake_array.kwargs == {'scheduler': 'synchronous'}

    with cube.use_dask_scheduler('threads'):
        cube._compute(fake_array)
        assert fake_array.kwargs == {'scheduler': 'threads'}

    cube._compute(fake_array)
    assert fake_array.kwargs == {'scheduler': 'synchronous'}

    cube.use_dask_scheduler('threads')
    cube._compute(fake_array)
    assert fake_array.kwargs == {'scheduler': 'threads'}

    with cube.use_dask_scheduler('processes', num_workers=4):
        cube._compute(fake_array)
        assert fake_array.kwargs == {'scheduler': 'processes', 'num_workers': 4}

    cube._compute(fake_array)
    assert fake_array.kwargs == {'scheduler': 'threads'}


def test_save_to_tmp_dir(data_adv):
    pytest.importorskip('zarr')
    cube = DaskSpectralCube.read(data_adv)
    cube_new = cube.sigma_clip_spectrally(3, save_to_tmp_dir=True)
    # The following test won't necessarily always work in future since the name
    # is not really guaranteed, but this is pragmatic enough for now
    assert cube_new._data.name.startswith('from-zarr')


def test_rechunk(data_adv):
    cube = DaskSpectralCube.read(data_adv)
    assert cube._data.chunksize == (4, 3, 2)
    cube_new = cube.rechunk(chunks=(1, 2, 3))
    # note last element is 2 because the chunk size we asked for
    # is larger than cube - this is fine and deliberate in this test
    assert cube_new._data.chunksize == (1, 2, 2)


def test_statistics(data_adv):
    cube = DaskSpectralCube.read(data_adv).rechunk(chunks=(1, 2, 3))
    stats = cube.statistics()
    assert_quantity_allclose(stats['npts'], 24)
    assert_quantity_allclose(stats['mean'], 0.4941651776136591 * u.K)
    assert_quantity_allclose(stats['sigma'], 0.3021908870982011 * u.K)
    assert_quantity_allclose(stats['sum'], 11.85996426272782 * u.K)
    assert_quantity_allclose(stats['sumsq'], 7.961125988022091 * u.K ** 2)
    assert_quantity_allclose(stats['min'], 0.0363300285196364 * u.K)
    assert_quantity_allclose(stats['max'], 0.9662900439556562 * u.K)
    assert_quantity_allclose(stats['rms'], 0.5759458158839716 * u.K)


@pytest.mark.skipif(not CASA_INSTALLED, reason='Requires CASA to be installed')
def test_statistics_consistency_casa(data_adv, tmp_path):

    # Similar to test_statistics but compares to CASA directly.

    cube = DaskSpectralCube.read(data_adv)
    stats = cube.statistics()

    make_casa_testimage(data_adv, tmp_path / 'casa.image')

    ia = casatools.image()
    ia.open(str(tmp_path / 'casa.image'))
    stats_casa = ia.statistics()
    ia.close()

    for key in stats:
        if isinstance(stats[key], u.Quantity):
            value = stats[key].value
        else:
            value = stats[key]
        assert_allclose(value, stats_casa[key])


def test_apply_function_parallel_spectral_noncube(data_adv):
    '''
    Testing returning a non-SpectralCube object with a user-defined
    function for spectral operations.
    '''

    chunk_size = (-1, 1, 2)
    cube = DaskSpectralCube.read(data_adv).rechunk(chunks=chunk_size)

    def sum_blocks_spectral(data_chunk):
        return data_chunk.sum(0)

    # Tell dask.map_blocks that we expect the zeroth axis to be (1,)
    output_chunk_size = (1, 2)

    test = cube.apply_function_parallel_spectral(sum_blocks_spectral,
                                                return_new_cube=False,
                                                accepts_chunks=True,
                                                drop_axis=[0], # The output will no longer contain the spectral axis
                                                chunks=output_chunk_size)

    # The total shape of test should be the (1,) + cube.shape[1:]
    assert test.shape == cube.shape[1:]

    # Test we get the same output as the builtin sum
    assert_allclose(test.compute(), cube.sum(axis=0).unitless_filled_data[:])


def test_apply_function_parallel_spectral_noncube_withblockinfo(data_adv):
    '''
    Test receiving block_info information from da.map_blocks so we can place
    the chunk's location in the whole cube when needed.

    https://docs.dask.org/en/latest/array-api.html#dask.array.map_blocks

    '''

    chunk_size = (-1, 1, 2)
    cube = DaskSpectralCube.read(data_adv).rechunk(chunks=chunk_size)

    sum_spectral_plane = cube.sum(axis=0).unitless_filled_data[:]
    # Each value should be different. This is important to check the right positions being used
    # for the check in sums_block_spectral
    assert np.unique(sum_spectral_plane).size == sum_spectral_plane.size


    def sum_blocks_spectral(data_chunk, block_info=None, comparison_array=None):

        chunk_sum = data_chunk.sum(0)

        # When the block_info kwarg is defined, it should not be None
        assert block_info is not None

        # Check the block location compared to `comparison_array`

        # Get the lower corner location in the whole cube.
        loc = [block_range[0] for block_range in block_info[0]['array-location']]
        # Should have 3 dimensions for the corner.
        assert len(loc) == 3

        # Slice comparison array to compare with this data chunk
        thisslice = (slice(loc[1], loc[1] + chunk_sum.shape[0]),
                     slice(loc[2], loc[2] + chunk_sum.shape[1]),)

        return chunk_sum == comparison_array[thisslice]

    # Tell dask.map_blocks that we expect the zeroth axis to be (1,)
    output_chunk_size = (1, 2)

    test = cube.apply_function_parallel_spectral(sum_blocks_spectral,
                                                return_new_cube=False,
                                                accepts_chunks=True,
                                                drop_axis=[0], # The output will no longer contain the spectral axis
                                                chunks=output_chunk_size,
                                                comparison_array=sum_spectral_plane) # Passed to `sum_blocks_spectral`

    # The total shape of test should be the (1,) + cube.shape[1:]
    assert test.shape == cube.shape[1:]

    # Test all True
    assert np.all(test.compute())


@pytest.mark.parametrize(('accepts_chunks'),
                         ((True, False)))
def test_apply_function_parallel_shape(accepts_chunks):
    # regression test for #772

    def func(x, add=None):
        if add is not None:
            y = x + add
        else:
            raise ValueError("This test is supposed to have add=1")
        return y


    fn = data.get_pkg_data_filename('tests/data/example_cube.fits', 'spectral_cube')
    cube = SpectralCube.read(fn, use_dask=True)
    cube2 = SpectralCube.read(fn, use_dask=False)

    # Check dask w/both threaded and unthreaded
    rslt3 = cube.apply_function_parallel_spectral(func, add=1,
                                                  accepts_chunks=accepts_chunks)
    with cube.use_dask_scheduler('threads', num_workers=4):
        rslt = cube.apply_function_parallel_spectral(func, add=1,
                                                     accepts_chunks=accepts_chunks)
    rslt2 = cube2.apply_function_parallel_spectral(func, add=1)

    np.testing.assert_almost_equal(cube.filled_data[:].value,
                                   cube2.filled_data[:].value)
    np.testing.assert_almost_equal(rslt.filled_data[:].value,
                                   rslt2.filled_data[:].value)
    np.testing.assert_almost_equal(rslt.filled_data[:].value,
                                   rslt3.filled_data[:].value)


@pytest.mark.parametrize('filename', ('data_adv', 'data_adv_beams',
    'data_vda_beams', 'data_vda_beams_image'))
def test_cube_on_cube(filename, request):
    if 'image' in filename and not CASA_INSTALLED:
        pytest.skip('Requires CASA to be installed')
    dataname = request.getfixturevalue(filename)

    # regression test for #782
    # the regression applies only to VaryingResolutionSpectralCubes
    # since they are not SpectralCube subclasses
    cube = DaskSpectralCube.read(dataname)
    assert isinstance(cube, (DaskSpectralCube, DaskVaryingResolutionSpectralCube))
    cube2 = SpectralCube.read(dataname, use_dask=False)
    if 'image' not in filename:
        # 'image' would be CASA and must be dask
        assert not isinstance(cube2, (DaskSpectralCube, DaskVaryingResolutionSpectralCube))

    with patch.object(cube, '_cube_on_cube_operation') as mock:
        cube * cube
    mock.assert_called_once()

    with patch.object(cube, '_cube_on_cube_operation') as mock:
        cube * cube2
    mock.assert_called_once()

    with patch.object(cube2, '_cube_on_cube_operation') as mock:
        cube2 * cube
    mock.assert_called_once()


if DISTRIBUTED_INSTALLED:

    def test_dask_distributed(client, tmpdir):  # noqa

        # Make sure that we can use dask distributed. This is a regression test for
        # a bug caused by FilledArrayHandler not being serializable.

        cube = DaskSpectralCube.read(os.path.join(DATA, 'basic.image'))
        cube.use_dask_scheduler(client)

        cube.sigma_clip_spectrally(2, save_to_tmp_dir=tmpdir.strpath)
