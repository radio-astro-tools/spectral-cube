from __future__ import print_function, absolute_import, division

import pytest
from distutils.version import LooseVersion

from .test_spectral_cube import cube_and_raw


@pytest.mark.openfiles_ignore
def test_projvis(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('matplotlib')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    mom0 = cube.moment0()
    mom0.quicklook(use_aplpy=False)


def test_proj_imshow(data_vda_jybeam_lower, use_dask):
    plt = pytest.importorskip('matplotlib.pyplot')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    mom0 = cube.moment0()
    if LooseVersion(plt.matplotlib.__version__) < LooseVersion('2.1'):
        # imshow is now only compatible with more recent versions of matplotlib
        # (apparently 2.0.2 was still incompatible)
        plt.imshow(mom0.value)
    else:
        plt.imshow(mom0)


@pytest.mark.openfiles_ignore
def test_projvis_aplpy(tmp_path, data_vda_jybeam_lower, use_dask):
    pytest.importorskip('aplpy')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    mom0 = cube.moment0()
    mom0.quicklook(use_aplpy=True, filename=tmp_path / 'test.png')


def test_mask_quicklook(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('aplpy')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    cube.mask.quicklook(view=(0, slice(None), slice(None)), use_aplpy=True)


@pytest.mark.openfiles_ignore
def test_to_glue(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('glue')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    cube.to_glue(start_gui=False)


@pytest.mark.openfiles_ignore
def test_to_pvextractor(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('pvextractor')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    cube.to_pvextractor()
