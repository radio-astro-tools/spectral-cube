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
    glue_data = app.data_collection[0]
    np.testing.assert_allclose(glue_data['SpectralCube'], cube.filled_data[:].value)

    # Add as a new component of an existing dataset
    cube.to_glue(dataset=glue_data)
    assert 'SpectralCube_' in [c.label for c in glue_data.main_components]
    app.close()


def test_to_glue_existing_app(data_vda_jybeam_lower, use_dask):
    glue_qt_app = pytest.importorskip('glue_qt.app')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    # Send to an already running glue session
    app = glue_qt_app.GlueApplication()
    cube.to_glue(glue_app=app)
    assert [d.label for d in app.data_collection] == ['SpectralCube']
    app.close()


def test_to_pvextractor(data_vda_jybeam_lower, use_dask):
    pytest.importorskip('pvextractor')
    cube, data = cube_and_raw(data_vda_jybeam_lower, use_dask=use_dask)
    cube.to_pvextractor()
