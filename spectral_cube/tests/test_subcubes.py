from __future__ import print_function, absolute_import, division

import pytest
from distutils.version import LooseVersion

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
    REGIONS_GT_03 = LooseVersion(regions.__version__) >= LooseVersion('0.3')
except ImportError:
    regionsOK = REGIONS_GT_03 = False

try:
    import scipy
    scipyOK = True
except ImportError:
    scipyOK = False


def test_subcube(data_advs, use_dask):

    cube, data = cube_and_raw(data_advs, use_dask=use_dask)

    sc1x = cube.subcube(xlo=1, xhi=3)
    sc2x = cube.subcube(xlo=24.06269*u.deg, xhi=24.06206*u.deg)
    sc2b = cube.subcube(xlo=24.06206*u.deg, xhi=24.06269*u.deg)
    # Mixed should be equivalent to above
    sc3x = cube.subcube(xlo=24.06269*u.deg, xhi=3)
    sc4x = cube.subcube(xlo=1, xhi=24.06206*u.deg)

    assert sc1x.shape == (2,3,2)
    assert sc2x.shape == (2,3,2)
    assert sc2b.shape == (2,3,2)
    assert sc3x.shape == (2,3,2)
    assert sc4x.shape == (2,3,2)
    assert sc1x.wcs.wcs.compare(sc2x.wcs.wcs)
    assert sc1x.wcs.wcs.compare(sc2b.wcs.wcs)
    assert sc1x.wcs.wcs.compare(sc3x.wcs.wcs)
    assert sc1x.wcs.wcs.compare(sc4x.wcs.wcs)

    sc1y = cube.subcube(ylo=1, yhi=3)
    sc2y = cube.subcube(ylo=29.93464 * u.deg,
                        yhi=29.93522 * u.deg)
    sc3y = cube.subcube(ylo=1, yhi=29.93522 * u.deg)
    sc4y = cube.subcube(ylo=29.93464 * u.deg, yhi=3)

    assert sc1y.shape == (2, 2, 4)
    assert sc2y.shape == (2, 2, 4)
    assert sc3y.shape == (2, 2, 4)
    assert sc4y.shape == (2, 2, 4)
    assert sc1y.wcs.wcs.compare(sc2y.wcs.wcs)
    assert sc1y.wcs.wcs.compare(sc3y.wcs.wcs)
    assert sc1y.wcs.wcs.compare(sc4y.wcs.wcs)


    # Test mixed slicing in both spatial directions
    sc1xy = cube.subcube(xlo=1, xhi=3, ylo=1, yhi=3)
    sc2xy = cube.subcube(xlo=24.06269*u.deg, xhi=3,
                         ylo=1,yhi=29.93522 * u.deg)

    sc3xy = cube.subcube(xlo=1, xhi=24.06206*u.deg,
                         ylo=29.93464 * u.deg, yhi=3)


    assert sc1xy.shape == (2, 2, 2)
    assert sc2xy.shape == (2, 2, 2)
    assert sc3xy.shape == (2, 2, 2)
    assert sc1xy.wcs.wcs.compare(sc2xy.wcs.wcs)
    assert sc1xy.wcs.wcs.compare(sc3xy.wcs.wcs)


    sc1z = cube.subcube(zlo=1, zhi=2)
    sc2z = cube.subcube(zlo=-320*u.km/u.s, zhi=-319*u.km/u.s)
    sc3z = cube.subcube(zlo=1, zhi=-319 * u.km / u.s)
    sc4z = cube.subcube(zlo=-320*u.km/u.s, zhi=2)
    assert sc1z.shape == (1, 3, 4)
    assert sc2z.shape == (1, 3, 4)
    assert sc3z.shape == (1, 3, 4)
    assert sc4z.shape == (1, 3, 4)
    assert sc1z.wcs.wcs.compare(sc2z.wcs.wcs)
    assert sc1z.wcs.wcs.compare(sc3z.wcs.wcs)
    assert sc1z.wcs.wcs.compare(sc4z.wcs.wcs)

    sc5 = cube.subcube()

    assert sc5.shape == cube.shape
    assert sc5.wcs.wcs.compare(cube.wcs.wcs)
    assert np.all(sc5._data == cube._data)


@pytest.mark.skipif('not scipyOK', reason='Could not import scipy')
@pytest.mark.skipif('not regionsOK', reason='Could not import regions')
@pytest.mark.skipif('not REGIONS_GT_03', reason='regions version should be >= 0.3')
@pytest.mark.parametrize('regfile',
                         ('255-fk5.reg', '255-pixel.reg'),
                        )
def test_ds9region_255(regfile, data_255, use_dask):
    # specific test for correctness
    cube, data = cube_and_raw(data_255, use_dask=use_dask)

    shapelist = regions.Regions.read(path(regfile))

    subcube = cube.subcube_from_regions(shapelist)
    assert_array_equal(subcube[0, :, :].value,
                           np.array([11, 12, 16, 17]).reshape((2, 2)))


@pytest.mark.skipif('not scipyOK', reason='Could not import scipy')
@pytest.mark.skipif('not regionsOK', reason='Could not import regions')
@pytest.mark.skipif('not REGIONS_GT_03', reason='regions version should be >= 0.3')
@pytest.mark.parametrize(('regfile', 'result'),
                             (('fk5.reg', (slice(None), 1, slice(None))),
                              ('fk5_twoboxes.reg', (slice(None), 1, slice(None))),
                              ('image.reg', (slice(None), 1, slice(None))),
                              (
                              'partial_overlap_image.reg', (slice(None), 1, 1)),
                              ('no_overlap_image.reg', ValueError),
                              ('partial_overlap_fk5.reg', (slice(None), 1, 1)),
                              ('no_overlap_fk5.reg', ValueError),
                              ))
def test_ds9region_new(regfile, result, data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)

    regionlist = regions.Regions.read(path(regfile))

    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result):
            sc = cube.subcube_from_regions(regionlist)
    else:
        sc = cube.subcube_from_regions(regionlist)

        # Shapes and size should be the same.
        # squeeze on the cube is b/c is retains dimensions of size 1
        assert sc.size == data[result].size
        assert sc.filled_data[:].squeeze().shape == data[result].shape

        # If sizes are the same, values should then be the same.
        assert (sc.unitless_filled_data[:].squeeze() == data[result]).all()

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
@pytest.mark.skipif('not REGIONS_GT_03', reason='regions version should be >= 0.3')
def test_regions_spectral(data_adv, use_dask):
    cube, data = cube_and_raw(data_adv, use_dask=use_dask)
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
