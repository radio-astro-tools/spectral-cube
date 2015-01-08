from ..io import class_lmv, fits
import pytest

@pytest.mark.skipif(True)
def test_lmv_fits():
    c1 = SpectralCube.read('example_cube.fits')
    c2 = SpectralCube.read('example_cube.lmv')
