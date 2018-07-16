from __future__ import print_function, absolute_import, division

import pytest

from astropy import units as u
from astropy import wcs
import numpy as np

from . import path
from .helpers import assert_allclose, assert_array_equal
from .test_spectral_cube import cube_and_raw
from ..spectral_axis import doppler_gamma, doppler_beta, doppler_z, get_rest_value_from_wcs

try:
    import regions
    regionsOK = True
except ImportError:
    regionsOK = False

try:
    import scipy
    scipyOK = True
except ImportError:
    scipyOK = False


def test_subcube():

    cube, data = cube_and_raw('advs.fits')

    sc1 = cube.subcube(xlo=1, xhi=3)
    sc2 = cube.subcube(xlo=24.06269*u.deg, xhi=24.06206*u.deg)
    sc2b = cube.subcube(xlo=24.06206*u.deg, xhi=24.06269*u.deg)

    assert sc1.shape == (2,3,2)
    assert sc2.shape == (2,3,2)
    assert sc2b.shape == (2,3,2)
    assert sc1.wcs.wcs.compare(sc2.wcs.wcs)
    assert sc1.wcs.wcs.compare(sc2b.wcs.wcs)

    sc3 = cube.subcube(ylo=1, yhi=3)
    sc4 = cube.subcube(ylo=29.93464 * u.deg,
                       yhi=29.93522 * u.deg)

    assert sc3.shape == (2, 2, 4)
    assert sc4.shape == (2, 2, 4)
    assert sc3.wcs.wcs.compare(sc4.wcs.wcs)

    sc5 = cube.subcube()

    assert sc5.shape == cube.shape
    assert sc5.wcs.wcs.compare(cube.wcs.wcs)
    assert np.all(sc5._data == cube._data)


@pytest.mark.skipif('not scipyOK', reason='Could not import scipy')
@pytest.mark.skipif('not regionsOK', reason='Could not import regions')
@pytest.mark.parametrize('regfile',
                         ('255-fk5.reg', '255-pixel.reg'),
                        )
def test_ds9region_255(regfile):
    # specific test for correctness
    cube, data = cube_and_raw('255.fits')

    shapelist = regions.read_ds9(path(regfile))

    subcube = cube.subcube_from_regions(shapelist)
    assert_array_equal(subcube[0, :, :].value,
                           np.array([11, 12, 16, 17]).reshape((2, 2)))


@pytest.mark.skipif('not scipyOK', reason='Could not import scipy')
@pytest.mark.skipif('not regionsOK', reason='Could not import regions')
@pytest.mark.parametrize(('regfile', 'result'),
                             (('fk5.reg', [slice(None), 1, 1]),
                              ('image.reg', [slice(None), 1, slice(None)]),
                              (
                              'partial_overlap_image.reg', [slice(None), 1, 1]),
                              ('no_overlap_image.reg', ValueError),
                              ('partial_overlap_fk5.reg', [slice(None), 1, 1]),
                              ('no_overlap_fk5.reg', ValueError),
                              ))
def test_ds9region_new(regfile, result):
    cube, data = cube_and_raw('adv.fits')

    regionlist = regions.read_ds9(path(regfile))

    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result) as exc:
            sc = cube.subcube_from_regions(regionlist)
        # this assertion is redundant, I think...
        assert exc.errisinstance(result)
    else:
        sc = cube.subcube_from_regions(regionlist)
        scsum = sc.sum()
        dsum = data[result].sum()
        assert_allclose(scsum, dsum)

    #region = 'fk5\ncircle(29.9346557, 24.0623827, 0.11111)'
    #subcube = cube.subcube_from_ds9region(region)
    # THIS TEST FAILS!
    # I think the coordinate transformation in ds9 is wrong;
    # it uses kapteyn?

    #region = 'circle(2,2,2)'
    #subcube = cube.subcube_from_ds9region(region)


@pytest.mark.skipif('not scipyOK', reason='Could not import scipy')
@pytest.mark.skipif('not regionsOK', reason='Could not import regions')
def test_regions_spectral():
    cube, data = cube_and_raw('adv.fits')
    rf_cube = get_rest_value_from_wcs(cube.wcs).to("GHz",
                                                         equivalencies=u.spectral())

    # content of image.reg
    regpix = regions.RectanglePixelRegion(regions.PixCoord(0.5, 1), width=4, height=2)

    # Velocity range in doppler_optical same as that of the cube.
    vel_range_optical = u.Quantity([-318 * u.km/u.s, -320 * u.km/u.s])
    regpix.meta['range'] = list(vel_range_optical)
    sc1 = cube.subcube_from_regions([regpix])
    scsum1 = sc1.sum()

    freq_range = vel_range_optical.to("GHz",
                                      equivalencies=u.doppler_optical(rf_cube))
    regpix.meta['range'] = list(freq_range)
    sc2 = cube.subcube_from_regions([regpix])
    scsum2 = sc2.sum()

    regpix.meta['restfreq'] = rf_cube
    vel_range_gamma = freq_range.to("km/s", equivalencies=doppler_gamma(rf_cube))
    regpix.meta['range'] = list(vel_range_gamma)
    regpix.meta['veltype'] = 'GAMMA'
    sc3 = cube.subcube_from_regions([regpix])
    scsum3 = sc3.sum()

    vel_range_beta = freq_range.to("km/s",
                                    equivalencies=doppler_beta(rf_cube))
    regpix.meta['range'] = list(vel_range_beta)
    regpix.meta['veltype'] = 'BETA'
    sc4 = cube.subcube_from_regions([regpix])
    scsum4 = sc4.sum()

    vel_range_z = freq_range.to("km/s",
                                    equivalencies=doppler_z(rf_cube))
    regpix.meta['range'] = list(vel_range_z)
    regpix.meta['veltype'] = 'Z'
    sc5 = cube.subcube_from_regions([regpix])
    scsum5 = sc5.sum()

    dsum = data[1:-1, 1, :].sum()
    assert_allclose(scsum1, dsum)
    # Proves that the vel/freq conversion works
    assert_allclose(scsum1, scsum2)
    assert_allclose(scsum2, scsum3)
    assert_allclose(scsum3, scsum4)
    assert_allclose(scsum4, scsum5)
