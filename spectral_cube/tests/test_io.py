from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.io import fits as pyfits
from astropy import units as u
from .. import SpectralCube, StokesSpectralCube
from ..lower_dimensional_structures import (OneDSpectrum,
                                            VaryingResolutionOneDSpectrum)
from . import path

from radio_beam import Beam

def test_lmv_fits():
    c1 = SpectralCube.read(path('example_cube.fits'))
    c2 = SpectralCube.read(path('example_cube.lmv'))

    assert c1.shape == c2.shape

    # should be able to do this, but it is not true:
    #assert c1.header == c2.header

    # should also be able to do this, but it is again false:
    #assert c1.wcs==c2.wcs


def test_3d_4d_stokes(data_adv, data_advs):
    f3 = pyfits.open(data_adv)
    f4 = pyfits.open(data_advs)
    f3b = pyfits.PrimaryHDU(data=f3[0].data, header=f4[0].header)

    c1 = SpectralCube.read(f3)
    c2 = SpectralCube.read(f4)
    c3 = SpectralCube.read(f3b)

    assert c1.shape == c3.shape
    # c2 has a different shape on disk...
    f3.close()
    f4.close()

def test_4d_stokes(data_advs):
    f = pyfits.open(data_advs)
    c = StokesSpectralCube.read(f)
    assert isinstance(c, StokesSpectralCube)
    f.close()


def test_4d_stokes_read_3d(data_adv):
    # Regression test for a bug that caused StokesSpectralCube.read to not work
    # correctly when reading in a 3D FITS file.
    f = pyfits.open(data_adv)
    c = StokesSpectralCube.read(f)
    assert isinstance(c, StokesSpectralCube)
    f.close()


def test_3d_beams(data_vda_beams):
    c = SpectralCube.read(data_vda_beams)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)


def test_4d_beams(data_sdav_beams):
    c = SpectralCube.read(data_sdav_beams)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)


def test_3d_beams_roundtrip(tmpdir, data_vda_beams):
    c = SpectralCube.read(data_vda_beams)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)
    c.write(tmpdir.join('vda_beams_out.fits').strpath)
    c2 = SpectralCube.read(tmpdir.join('vda_beams_out.fits').strpath)

    # assert c==c2 # this is not implemented?
    assert np.all(c.filled_data[:] == c2.filled_data[:])
    #assert c.wcs == c2.wcs # not implemented correctly?

    np.testing.assert_almost_equal(c2.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c2.beams[0].minor.value, 0.1)


def test_4d_beams_roundtrip(tmpdir, data_sdav_beams):
    # not sure if 4d can round-trip...
    c = SpectralCube.read(data_sdav_beams)
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c.beams[0].minor.value, 0.1)
    c.write(tmpdir.join('sdav_beams_out.fits').strpath)
    c2 = SpectralCube.read(tmpdir.join('sdav_beams_out.fits').strpath)

    # assert c==c2 # this is not implemented?
    assert np.all(c.filled_data[:] == c2.filled_data[:])
    #assert c.wcs == c2.wcs # not implemented correctly?

    np.testing.assert_almost_equal(c2.beams[0].major.value, 0.4)
    np.testing.assert_almost_equal(c2.beams[0].minor.value, 0.1)

def test_1d(data_5_spectral):
    hdul = pyfits.open(data_5_spectral)
    hdu = hdul[0]
    spec = OneDSpectrum.from_hdu(hdu)

    np.testing.assert_almost_equal(spec, np.arange(5, dtype='float'))
    hdul.close()

def test_1d_beams(data_5_spectral_beams):
    hdu = pyfits.open(data_5_spectral_beams)
    spec = OneDSpectrum.from_hdu(hdu)

    np.testing.assert_almost_equal(spec, np.arange(5, dtype='float'))
    assert isinstance(spec, VaryingResolutionOneDSpectrum)
    assert hasattr(spec, 'beams')
    assert len(spec.beams) == 5
    hdu.close()
