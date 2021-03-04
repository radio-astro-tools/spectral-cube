# Tests specific to the dask class

import pytest

from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as u

from spectral_cube import DaskSpectralCube
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
