from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.io import fits as pyfits
from astropy import units as u
from .. import StokesSpectralCube
from ..lower_dimensional_structures import (OneDSpectrum,
                                            VaryingResolutionOneDSpectrum)
from . import path

from radio_beam import Beam


from .. import SpectralCube, VaryingResolutionSpectralCube
from ..dask_spectral_cube import DaskSpectralCube


def test_lmv_fits():
    c1 = SpectralCube.read(path('example_cube.fits'))
    c2 = SpectralCube.read(path('example_cube.lmv'))

    assert c1.shape == c2.shape

    # should be able to do this, but it is not true:
    #assert c1.header == c2.header

    # should also be able to do this, but it is again false:
    #assert c1.wcs==c2.wcs


def test_3d_4d_stokes(data_adv, data_advs, use_dask):
    f3 = pyfits.open(data_adv)
    f4 = pyfits.open(data_advs)
    f3b = pyfits.PrimaryHDU(data=f3[0].data, header=f4[0].header)

    c1 = SpectralCube.read(f3, use_dask=use_dask)
    c2 = SpectralCube.read(f4, use_dask=use_dask)
    c3 = SpectralCube.read(f3b, use_dask=use_dask)

    assert c1.shape == c3.shape
    # c2 has a different shape on disk...
    f3.close()
    f4.close()


def test_4d_stokes(data_advs, use_dask):
    f = pyfits.open(data_advs)
    c = StokesSpectralCube.read(f, use_dask=use_dask)
    assert isinstance(c, StokesSpectralCube)
    if use_dask:
        assert isinstance(c.I, DaskSpectralCube)
    else:
        assert isinstance(c.I, SpectralCube)
    f.close()


def test_4d_stokes_read_3d(data_adv, use_dask):
    # Regression test for a bug that caused StokesSpectralCube.read to not work
    # correctly when reading in a 3D FITS file.
    f = pyfits.open(data_adv)
    c = StokesSpectralCube.read(f, use_dask=use_dask)
    assert isinstance(c, StokesSpectralCube)
    f.close()


def test_3d_beams(data_vda_beams, use_dask):
    c = SpectralCube.read(data_vda_beams, use_dask=use_dask)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)


def test_4d_beams(data_sdav_beams, use_dask):
    c = SpectralCube.read(data_sdav_beams, use_dask=use_dask)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)


def test_4d_beams_nounits(data_sdav_beams_nounits, use_dask):
    c = SpectralCube.read(data_sdav_beams_nounits, use_dask=use_dask)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)
    assert c.beams[0].major.unit == u.arcsec
    assert c.beams[0].minor.unit == u.arcsec


def test_3d_beams_roundtrip(tmpdir, data_vda_beams, use_dask):
    c = SpectralCube.read(data_vda_beams, use_dask=use_dask)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)
    c.write(tmpdir.join('vda_beams_out.fits').strpath)
    c2 = SpectralCube.read(tmpdir.join('vda_beams_out.fits').strpath, use_dask=use_dask)

    # assert c==c2 # this is not implemented?
    assert np.all(c.filled_data[:] == c2.filled_data[:])
    #assert c.wcs == c2.wcs # not implemented correctly?

    np.testing.assert_almost_equal(c2.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c2.beams[0].minor.value, 0.1)
    assert c2.beams[0].major.unit == u.arcsec
    assert c2.beams[0].minor.unit == u.arcsec


def test_4d_beams_roundtrip(tmpdir, data_sdav_beams, use_dask):
    # not sure if 4d can round-trip...
    c = SpectralCube.read(data_sdav_beams, use_dask=use_dask)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)
    c.write(tmpdir.join('sdav_beams_out.fits').strpath)
    c2 = SpectralCube.read(tmpdir.join('sdav_beams_out.fits').strpath, use_dask=use_dask)

    # assert c==c2 # this is not implemented?
    assert np.all(c.filled_data[:] == c2.filled_data[:])
    #assert c.wcs == c2.wcs # not implemented correctly?

    np.testing.assert_almost_equal(c2.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c2.beams[0].minor.value, 0.1)
    assert c2.beams[0].major.unit == u.arcsec
    assert c2.beams[0].minor.unit == u.arcsec


def test_1d(data_5_spectral):
    hdul = pyfits.open(data_5_spectral)
    hdu = hdul[0]
    spec = OneDSpectrum.from_hdu(hdu)

    np.testing.assert_almost_equal(spec, np.arange(5, dtype='float'))
    hdul.close()

def test_1d_beams(data_5_spectral_beams, use_dask):
    hdu = pyfits.open(data_5_spectral_beams)
    spec = OneDSpectrum.from_hdu(hdu)

    np.testing.assert_almost_equal(spec, np.arange(5, dtype='float'))
    assert isinstance(spec, VaryingResolutionOneDSpectrum)
    assert hasattr(spec, 'beams')
    assert len(spec.beams) == 5
    hdu.close()



def test_aips_beams_units(tmpdir, data_455_degree_beams, use_dask):
    """
    regression test for #737, AIPS beam unit specs (degrees)
    """
    c = SpectralCube.read(data_455_degree_beams, use_dask=use_dask)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4/3600)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1/3600)
    np.testing.assert_almost_equal(c.beams[0].major.to(u.arcsec).value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.to(u.arcsec).value, 0.1)



def test_vrsc_fullstokes_read_fits(data_advs_beams_fullstokes):
    '''
    Test reading a spectral line data cube with full stokes and a beam table.
    '''
    c = StokesSpectralCube.read(data_advs_beams_fullstokes)

    for component in ['I', 'Q', 'U', 'V']:
        assert isinstance(c[component], VaryingResolutionSpectralCube)
        assert hasattr(c[component], 'beams')
