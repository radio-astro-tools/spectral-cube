from __future__ import print_function, absolute_import, division

import numpy as np
from astropy import units as u

from .helpers import assert_allclose
from ..lower_dimensional_structures import Projection

def test_slices_of_projections_not_projections():
    # slices of projections that have <2 dimensions should not be projections
    image = np.ones((2, 2)) * u.Jy
    p = Projection(image, copy=False)

    assert not isinstance(p[0,0], Projection)
    assert not isinstance(p[0], Projection)


def test_copy_false():
    image = np.ones((12, 12)) * u.Jy
    p = Projection(image, copy=False)
    image[3,4] = 2 * u.Jy
    assert_allclose(p[3,4], 2 * u.Jy)

def test_write(tmpdir):
    image = np.ones((12, 12)) * u.Jy
    p = Projection(image)
    p.write(tmpdir.join('test.fits').strpath)

def test_preserve_wcs_to():
    # regression for #256
    image = np.ones((12, 12)) * u.Jy
    p = Projection(image, copy=False)
    image[3,4] = 2 * u.Jy

    p2 = p.to(u.mJy)
    assert_allclose(p[3,4], 2 * u.Jy)
    assert_allclose(p[3,4], 2000 * u.mJy)

    assert p2.wcs == p.wcs

def test_multiplication():
    # regression: 265

    image = np.ones((12, 12)) * u.Jy
    p = Projection(image, copy=False)

    p2 = p * 5

    assert p2.unit == u.Jy
    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value == 5)

def test_unit_division():
    # regression: 265

    image = np.ones((12, 12)) * u.Jy
    p = Projection(image, copy=False)

    p2 = p / u.beam

    assert p2.unit == u.Jy/u.beam
    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs

def test_isnan():
    # Check that np.isnan strips units

    image = np.ones((12, 12)) * u.Jy
    image[5,6] = np.nan
    p = Projection(image, copy=False)

    mask = np.isnan(p)

    assert mask.sum() == 1
    assert not hasattr(mask, 'unit')
