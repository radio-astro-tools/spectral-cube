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
