import numpy as np
from astropy import units as u

from .helpers import assert_allclose
from ..spectral_cube import Projection


def test_copy_false():
    image = np.ones((12, 12)) * u.Jy
    p = Projection(image, copy=False)
    image[3,4] = 2 * u.Jy
    assert_allclose(p[3,4], 2 * u.Jy)


def test_write(tmpdir):
    image = np.ones((12, 12)) * u.Jy
    p = Projection(image)
    p.write(tmpdir.join('test.fits').strpath)
