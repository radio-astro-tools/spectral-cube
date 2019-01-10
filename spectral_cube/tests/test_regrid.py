import pytest
import numpy as np

from astropy import units as u
from astropy import convolution
from astropy.wcs import WCS
from astropy import wcs
from astropy.io import fits

from radio_beam import beam, Beam

from .. import SpectralCube
from ..utils import WCSCelestialError
from .test_spectral_cube import cube_and_raw
from .test_projection import load_projection
from . import path


def test_convolution():
    cube, data = cube_and_raw('255_delta.fits')

    # 1" convolved with 1.5" -> 1.8027....
    target_beam = Beam(1.802775637731995*u.arcsec, 1.802775637731995*u.arcsec,
                       0*u.deg)

    conv_cube = cube.convolve_to(target_beam)

    expected = convolution.Gaussian2DKernel((1.5*u.arcsec /
                                             beam.SIGMA_TO_FWHM /
                                             (5.555555555555e-4*u.deg)).decompose().value,
                                            x_size=5, y_size=5,
                                           )

    expected.normalize()

    np.testing.assert_almost_equal(expected.array,
                                   conv_cube.filled_data[0,:,:].value)

    # 2nd layer is all zeros
    assert np.all(conv_cube.filled_data[1,:,:] == 0.0)


def test_beams_convolution():
    cube, data = cube_and_raw('455_delta_beams.fits')

    # 1" convolved with 1.5" -> 1.8027....
    target_beam = Beam(1.802775637731995*u.arcsec, 1.802775637731995*u.arcsec,
                       0*u.deg)

    conv_cube = cube.convolve_to(target_beam)

    pixscale = wcs.utils.proj_plane_pixel_area(cube.wcs.celestial)**0.5*u.deg

    for ii, bm in enumerate(cube.beams):
        expected = target_beam.deconvolve(bm).as_kernel(pixscale, x_size=5,
                                                        y_size=5)
        expected.normalize()

        np.testing.assert_almost_equal(expected.array,
                                       conv_cube.filled_data[ii,:,:].value)


def test_beams_convolution_equal():
    cube, data = cube_and_raw('522_delta_beams.fits')

    # Only checking that the equal beam case is handled correctly.
    # Fake the beam in the first channel. Then ensure that the first channel
    # has NOT been convolved.
    target_beam = Beam(1.0 * u.arcsec, 1.0 * u.arcsec, 0.0 * u.deg)
    cube.beams.major[0] = target_beam.major
    cube.beams.minor[0] = target_beam.minor
    cube.beams.pa[0] = target_beam.pa

    conv_cube = cube.convolve_to(target_beam)

    np.testing.assert_almost_equal(cube.filled_data[0].value,
                                   conv_cube.filled_data[0].value)


@pytest.mark.parametrize('use_memmap', (True, False))
def test_reproject(use_memmap):

    pytest.importorskip('reproject')

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

    result = cube.reproject(header_out, use_memmap=use_memmap)

    assert result.shape == (cube.shape[0], 5, 4)


def test_spectral_smooth():

    cube, data = cube_and_raw('522_delta.fits')

    result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(1.0), use_memmap=False)

    np.testing.assert_almost_equal(result[:,0,0].value,
                                   convolution.Gaussian1DKernel(1.0,
                                                                x_size=5).array,
                                   4)

    result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(1.0), use_memmap=True)

    np.testing.assert_almost_equal(result[:,0,0].value,
                                   convolution.Gaussian1DKernel(1.0,
                                                                x_size=5).array,
                                   4)


def test_spectral_smooth_4cores():

    pytest.importorskip('joblib')

    cube, data = cube_and_raw('522_delta.fits')

    result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(1.0), num_cores=4, use_memmap=True)

    np.testing.assert_almost_equal(result[:,0,0].value,
                                   convolution.Gaussian1DKernel(1.0,
                                                                x_size=5).array,
                                   4)

    # this is one way to test non-parallel mode
    result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(1.0), num_cores=4, use_memmap=False)

    np.testing.assert_almost_equal(result[:,0,0].value,
                                   convolution.Gaussian1DKernel(1.0,
                                                                x_size=5).array,
                                   4)

    # num_cores = 4 is a contradiction with parallel=False, but we want to make
    # sure it does the same thing
    result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(1.0), num_cores=4, parallel=False)

    np.testing.assert_almost_equal(result[:,0,0].value,
                                   convolution.Gaussian1DKernel(1.0,
                                                                x_size=5).array,
                                   4)


def test_spectral_smooth_fail():

    cube, data = cube_and_raw('522_delta_beams.fits')

    with pytest.raises(AttributeError) as exc:
        cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(1.0))

    assert exc.value.args[0] == ("VaryingResolutionSpectralCubes can't be "
                                 "spectrally smoothed.  Convolve to a "
                                 "common resolution with `convolve_to` before "
                                 "attempting spectral smoothed.")


def test_spectral_interpolate():

    cube, data = cube_and_raw('522_delta.fits')

    orig_wcs = cube.wcs.deepcopy()

    # midpoint between each position
    sg = (cube.spectral_axis[1:] + cube.spectral_axis[:-1])/2.

    result = cube.spectral_interpolate(spectral_grid=sg)

    np.testing.assert_almost_equal(result[:,0,0].value,
                                   [0.0, 0.5, 0.5, 0.0])

    assert cube.wcs.wcs.compare(orig_wcs.wcs)


def test_spectral_interpolate_with_fillvalue():

    cube, data = cube_and_raw('522_delta.fits')

    # Step one channel out of bounds.
    sg = ((cube.spectral_axis[0]) -
          (cube.spectral_axis[1] - cube.spectral_axis[0]) *
          np.linspace(1,4,4))
    result = cube.spectral_interpolate(spectral_grid=sg,
                                       fill_value=42)
    np.testing.assert_almost_equal(result[:,0,0].value,
                                   np.ones(4)*42)


def test_spectral_interpolate_fail():

    cube, data = cube_and_raw('522_delta_beams.fits')

    with pytest.raises(AttributeError) as exc:
        cube.spectral_interpolate(5)

    assert exc.value.args[0] == ("VaryingResolutionSpectralCubes can't be "
                                 "spectrally interpolated.  Convolve to a "
                                 "common resolution with `convolve_to` before "
                                 "attempting spectral interpolation.")


def test_spectral_interpolate_with_mask():

    hdu = fits.open(path("522_delta.fits"))[0]

    # Swap the velocity axis so indiff < 0 in spectral_interpolate
    hdu.header["CDELT3"] = - hdu.header["CDELT3"]

    cube = SpectralCube.read(hdu)

    mask = np.ones(cube.shape, dtype=bool)
    mask[:2] = False

    masked_cube = cube.with_mask(mask)

    orig_wcs = cube.wcs.deepcopy()

    # midpoint between each position
    sg = (cube.spectral_axis[1:] + cube.spectral_axis[:-1])/2.

    result = masked_cube.spectral_interpolate(spectral_grid=sg[::-1])

    # The output makes CDELT3 > 0 (reversed spectral axis) so the masked
    # portion are the final 2 channels.
    np.testing.assert_almost_equal(result[:, 0, 0].value,
                                   [0.0, 0.5, np.NaN, np.NaN])

    assert cube.wcs.wcs.compare(orig_wcs.wcs)


def test_spectral_interpolate_reversed():

    cube, data = cube_and_raw('522_delta.fits')

    orig_wcs = cube.wcs.deepcopy()

    # Reverse spectral axis
    sg = cube.spectral_axis[::-1]

    result = cube.spectral_interpolate(spectral_grid=sg)

    np.testing.assert_almost_equal(sg.value, result.spectral_axis.value)


def test_convolution_2D():

    proj, hdu = load_projection("55_delta.fits")

    # 1" convolved with 1.5" -> 1.8027....
    target_beam = Beam(1.802775637731995*u.arcsec, 1.802775637731995*u.arcsec,
                       0*u.deg)

    conv_proj = proj.convolve_to(target_beam)

    expected = convolution.Gaussian2DKernel((1.5*u.arcsec /
                                             beam.SIGMA_TO_FWHM /
                                             (5.555555555555e-4*u.deg)).decompose().value,
                                            x_size=5, y_size=5,
                                           )
    expected.normalize()

    np.testing.assert_almost_equal(expected.array,
                                   conv_proj.value)
    assert conv_proj.beam == target_beam



def test_nocelestial_convolution_2D_fail():

    cube, data = cube_and_raw('255_delta.fits')

    proj = cube.moment0(axis=1)

    test_beam = Beam(1.0 * u.arcsec)

    with pytest.raises(WCSCelestialError) as exc:
        proj.convolve_to(test_beam)

    assert exc.value.args[0] == ("WCS does not contain two spatial axes.")


def test_reproject_2D():

    pytest.importorskip('reproject')

    proj, hdu = load_projection("55.fits")

    wcs_in = WCS(proj.header)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN']
    wcs_out.wcs.crval = [134.37608, -31.939241]
    wcs_out.wcs.crpix = [2., 2.]

    header_out = proj.header
    header_out['NAXIS1'] = 4
    header_out['NAXIS2'] = 5
    header_out.update(wcs_out.to_header())

    result = proj.reproject(header_out)

    assert result.shape == (5, 4)
    assert result.beam == proj.beam


def test_nocelestial_reproject_2D_fail():

    pytest.importorskip('reproject')

    cube, data = cube_and_raw('255_delta.fits')

    proj = cube.moment0(axis=1)

    with pytest.raises(WCSCelestialError) as exc:
        proj.reproject(cube.header)

    assert exc.value.args[0] == ("WCS does not contain two spatial axes.")


@pytest.mark.parametrize('use_memmap', (True,False))
def test_downsample(use_memmap):
    cube, data = cube_and_raw('255.fits')

    dscube = cube.downsample_axis(factor=2, axis=0, use_memmap=use_memmap)

    expected = data.mean(axis=0)

    np.testing.assert_almost_equal(expected[None,:,:],
                                   dscube.filled_data[:].value)

    dscube = cube.downsample_axis(factor=2, axis=1, use_memmap=use_memmap)

    expected = np.array([data[:,:2,:].mean(axis=1),
                         data[:,2:4,:].mean(axis=1),
                         data[:,4:,:].mean(axis=1), # just data[:,4,:]
                        ]).swapaxes(0,1)
    assert expected.shape == (2,3,5)
    assert dscube.shape == (2,3,5)

    np.testing.assert_almost_equal(expected,
                                   dscube.filled_data[:].value)

    dscube = cube.downsample_axis(factor=2, axis=1, truncate=True,
                                  use_memmap=use_memmap)

    expected = np.array([data[:,:2,:].mean(axis=1),
                         data[:,2:4,:].mean(axis=1),
                        ]).swapaxes(0,1)

    np.testing.assert_almost_equal(expected,
                                   dscube.filled_data[:].value)
