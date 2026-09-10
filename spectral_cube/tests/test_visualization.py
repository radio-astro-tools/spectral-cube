import numpy as np
import pytest

from .test_spectral_cube import cube_and_raw


def test_projvis(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('matplotlib')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    mom0 = cube.moment0()
    mom0.quicklook(use_aplpy=False)


def test_proj_imshow(data_vda_jybeam_lower, use_dask):
    plt = pytest.importorskip('matplotlib.pyplot')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    mom0 = cube.moment0()
    plt.imshow(mom0)


def test_projvis_aplpy(tmp_path, data_vda_jybeam_lower, use_dask):
    pytest.importorskip('aplpy')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    mom0 = cube.moment0()
    mom0.quicklook(use_aplpy=True, filename=tmp_path / 'test.png')


def test_mask_quicklook(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('aplpy')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    cube.mask.quicklook(view=(0, slice(None), slice(None)), use_aplpy=True)


def test_to_glue(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('glue_qt')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    app = cube.to_glue(start_gui=False)
    try:
        expected = cube.filled_data[:].value
        glue_data = app.data_collection[0]
        np.testing.assert_allclose(glue_data['SpectralCube'], expected)

        # Add as a new component of an existing dataset
        cube.to_glue(dataset=glue_data)
        np.testing.assert_allclose(glue_data['SpectralCube_'], expected)
    finally:
        app.close()


def test_to_glue_existing_app(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('glue_qt')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    app = cube.to_glue(start_gui=False)
    try:
        # A cube that has not been sent to glue before
        cube2, _ = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
        cube2.to_glue(name='cube2', glue_app=app, start_gui=False)
        assert [d.label for d in app.data_collection] == ['SpectralCube', 'cube2']
    finally:
        app.close()


def test_to_pvextractor(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('pvextractor')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    cube.to_pvextractor()
