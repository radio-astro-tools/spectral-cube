import pytest
import numpy as np
from astropy import units as u
from astropy import convolution

from .test_spectral_cube import cube_and_raw

try:
    from radio_beam import beam,Beam
    RADIO_BEAM_INSTALLED = True
except ImportError:
    RADIO_BEAM_INSTALLED = False

try:
    import reproject
    REPROJECT_INSTALLED = True
except ImportError:
    REPROJECT_INSTALLED = False


@pytest.mark.skipif('not RADIO_BEAM_INSTALLED')
def test_convolution():
    cube, data = cube_and_raw('255_delta.fits')

    # 1" convolved with 1.5" -> 1.8027....
    target_beam = Beam(1.802775637731995*u.arcsec, 1.802775637731995*u.arcsec,
                       0*u.deg)

    conv_cube = cube.convolve_to(target_beam)

    expected = convolution.Gaussian2DKernel((1.5*u.arcsec /
                                             beam.SIGMA_TO_FWHM /
                                             (5.555555555555e-4*u.deg)).decompose().value,
                                           x_size=5,
                                           y_size=5,
                                           )

    np.testing.assert_almost_equal(expected.array,
                                   conv_cube.filled_data[0,:,:].value)

    # 2nd layer is all zeros
    assert np.all(conv_cube.filled_data[1,:,:] == 0.0)


@pytest.mark.skipif('not REPROJECT_INSTALLED')
def test_reproject():

    cube, data = cube_and_raw('adv.fits')

    wcs_in = WCS(cube.header)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [134.37608, -31.939241, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [2., 2., wcs_in.wcs.crpix[2]]

    header_out = cube.header
    header_out['NAXIS1'] = 4
    header_out['NAXIS2'] = 5
    header_out['NAXIS3'] = cube.shape[0]
    header_out.update(wcs_out.to_header())

    result = cube.reproject(header_out)

    assert result.shape == (cube.shape[0], 5, 4)
