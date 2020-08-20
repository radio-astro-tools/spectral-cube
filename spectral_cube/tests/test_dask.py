# Tests specific to the dask class

import pytest

from spectral_cube import DaskSpectralCube


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
