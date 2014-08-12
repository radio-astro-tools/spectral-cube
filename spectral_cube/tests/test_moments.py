import pytest
import numpy as np

from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits

from ..spectral_cube import SpectralCube
from .helpers import assert_allclose

# the back of the book
dv = 3e-2 * u.Unit('m/s')
dy = 2e-5 * u.Unit('deg')
dx = 1e-5 * u.Unit('deg')
data_unit = u.dimensionless_unscaled

m0v = np.array([[27, 30, 33],
                [36, 39, 42],
                [45, 48, 51]]) * data_unit * dv
m0y = np.array([[9, 12, 15],
                [36, 39, 42],
                [63, 66, 69]]) * data_unit * dy
m0x = np.array([[3, 12, 21],
                [30, 39, 48],
                [57, 66, 75]]) * data_unit * dx

# M1V is a special case, where we return the actual coordinate
m1v = np.array([[1.66666667, 1.6, 1.54545455],
                [1.5, 1.46153846, 1.42857143],
                [1.4, 1.375, 1.35294118]]) * dv + 2 * u.Unit('m/s')
m1y = np.array([[1.66666667, 1.5, 1.4],
                [1.16666667, 1.15384615, 1.14285714],
                [1.0952381, 1.09090909, 1.08695652]]) * dy
m1x = np.array([[1.66666667, 1.16666667, 1.0952381],
                [1.06666667, 1.05128205, 1.04166667],
                [1.03508772, 1.03030303, 1.02666667]]) * dx

m2v = np.array([[0.22222222, 0.30666667, 0.36914601],
               [0.41666667, 0.45364892, 0.4829932],
               [0.50666667, 0.52604167, 0.54209919]]) * dv ** 2
m2y = np.array([[0.22222222, 0.41666667, 0.50666667],
                [0.63888889, 0.64299803, 0.6462585],
                [0.65759637, 0.6584022, 0.65910523]]) * dy ** 2
m2x = np.array([[0.22222222, 0.63888889, 0.65759637],
                [0.66222222, 0.66403682, 0.66493056],
                [0.66543552, 0.66574839, 0.66595556]]) * dx ** 2
MOMENTS = [[m0v, m0y, m0x], [m1v, m1y, m1x], [m2v, m2y, m2x]]


def moment_cube():
    cube = np.arange(27).reshape([3, 3, 3]).astype(np.float)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VELO']
    # choose values to minimize spherical distortions
    wcs.wcs.cdelt = np.array([-1, 2, 3], dtype='float32') / 1e5
    wcs.wcs.crpix = np.array([1, 1, 1], dtype='float32')
    wcs.wcs.crval = np.array([0, 1e-3, 2e-3], dtype='float32')
    wcs.wcs.cunit = ['deg', 'deg', 'km/s']

    hdu = fits.PrimaryHDU(data=cube, header=wcs.to_header())
    return hdu

axis_order = pytest.mark.parametrize(('axis', 'order'),
                                    ((0, 0), (0, 1), (0, 2),
                                     (1, 0), (1, 1), (1, 2),
                                     (2, 0), (2, 1), (2, 2)))


@axis_order
def test_strategies_consistent(axis, order):
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)

    cwise = sc.moment(axis=axis, order=order, how='cube')
    swise = sc.moment(axis=axis, order=order, how='slice')
    rwise = sc.moment(axis=axis, order=order, how='ray')
    assert_allclose(cwise, swise)
    assert_allclose(cwise, rwise)


@pytest.mark.parametrize(('order', 'axis', 'how'),
                         [(o, a, h)
                          for o in [0, 1, 2]
                          for a in [0, 1, 2]
                          for h in ['cube', 'slice', 'auto', 'ray']])
def test_reference(order, axis, how):
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)
    mom_sc = sc.moment(order=order, axis=axis, how=how)
    assert_allclose(mom_sc, MOMENTS[order][axis])


@axis_order
def test_consistent_mask_handling(axis, order):
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)
    sc._mask = sc > 4

    cwise = sc.moment(axis=axis, order=order, how='cube')
    swise = sc.moment(axis=axis, order=order, how='slice')
    rwise = sc.moment(axis=axis, order=order, how='ray')
    assert_allclose(cwise, swise)
    assert_allclose(cwise, rwise)


def test_convenience_methods():
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)

    assert_allclose(sc.moment0(axis=0), MOMENTS[0][0])
    assert_allclose(sc.moment1(axis=2), MOMENTS[1][2])
    assert_allclose(sc.moment2(axis=1), MOMENTS[2][1])

def test_preserve_unit():
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)
    sc_kms = sc.with_spectral_unit(u.km/u.s)
    m0 = sc_kms.moment0(axis=0)
    m1 = sc_kms.moment1(axis=0)

    assert_allclose(m0, MOMENTS[0][0].to(u.km/u.s))
    assert_allclose(m1, MOMENTS[1][0].to(u.km/u.s))
