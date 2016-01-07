from __future__ import print_function, absolute_import, division

from astropy.io import fits as pyfits
from ..io import class_lmv, fits
from .. import SpectralCube
from . import path
import pytest

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
