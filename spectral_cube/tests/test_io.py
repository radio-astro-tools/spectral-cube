from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.io import fits as pyfits
from astropy import units as u
from ..io import class_lmv, fits
from .. import SpectralCube, StokesSpectralCube
from . import path
import pytest

from radio_beam import Beam

def test_lmv_fits():
    c1 = SpectralCube.read(path('example_cube.fits'))
    c2 = SpectralCube.read(path('example_cube.lmv'))

    assert c1.shape == c2.shape

    # should be able to do this, but it is not true:
    #assert c1.header == c2.header

    # should also be able to do this, but it is again false:
    #assert c1.wcs==c2.wcs

def test_3d_4d_stokes():
    f3 = pyfits.open(path('adv.fits'))
    f4 = pyfits.open(path('advs.fits'))
    f3b = pyfits.PrimaryHDU(data=f3[0].data, header=f4[0].header)

    c1 = SpectralCube.read(f3)
    c2 = SpectralCube.read(f4)
    c3 = SpectralCube.read(f3b)

    assert c1.shape == c3.shape
    # c2 has a different shape on disk...

def test_4d_stokes():
    f = pyfits.open(path('advs.fits'))
    c = StokesSpectralCube.read(f)
    assert isinstance(c, StokesSpectralCube)

def test_3d_beams():
    c = SpectralCube.read(path('vda_beams.fits'))
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.1)

def test_4d_beams():
    c = SpectralCube.read(path('sdav_beams.fits'))
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.1)


def test_3d_beams_roundtrip():
    c = SpectralCube.read(path('vda_beams.fits'))
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.1)
    c.write(path('vda_beams_out.fits'))
    c2 = SpectralCube.read(path('vda_beams_out.fits'))

    # assert c==c2 # this is not implemented?
    assert np.all(c.filled_data[:] == c2.filled_data[:])
    #assert c.wcs == c2.wcs # not implemented correctly?

    np.testing.assert_almost_equal(c2.beams[0].major.value, 0.1)


def test_4d_beams_roundtrip():
    # not sure if 4d can round-trip...
    c = SpectralCube.read(path('sdav_beams.fits'))
    np.testing.assert_almost_equal(c.beams[0].major.value, 0.1)
    c.write(path('sdav_beams_out.fits'))
    c2 = SpectralCube.read(path('sdav_beams_out.fits'))

    # assert c==c2 # this is not implemented?
    assert np.all(c.filled_data[:] == c2.filled_data[:])
    #assert c.wcs == c2.wcs # not implemented correctly?

    np.testing.assert_almost_equal(c2.beams[0].major.value, 0.1)
