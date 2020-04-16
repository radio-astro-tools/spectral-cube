
'''
Utilities for tests.
'''

from six.moves import zip

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.utils import NumpyRNGContext

from ..spectral_cube import SpectralCube


def generate_header(pixel_scale, spec_scale, beamfwhm, imshape, v0):

    header = {'CDELT1': -(pixel_scale).to(u.deg).value,
              'CDELT2': (pixel_scale).to(u.deg).value,
              'BMAJ': beamfwhm.to(u.deg).value,
              'BMIN': beamfwhm.to(u.deg).value,
              'BPA': 0.0,
              'CRPIX1': imshape[0] / 2.,
              'CRPIX2': imshape[1] / 2.,
              'CRVAL1': 0.0,
              'CRVAL2': 0.0,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CUNIT1': 'deg',
              'CUNIT2': 'deg',
              'CRVAL3': v0,
              'CUNIT3': spec_scale.unit.to_string(),
              'CDELT3': spec_scale.value,
              'CRPIX3': 1,
              'CTYPE3': 'VRAD',
              'BUNIT': 'K',
              }

    return fits.Header(header)


def generate_hdu(data, pixel_scale, spec_scale, beamfwhm, v0):

    imshape = data.shape[1:]

    header = generate_header(pixel_scale, spec_scale, beamfwhm, imshape, v0)

    return fits.PrimaryHDU(data, header)


def gaussian(x, amp, mean, sigma):
    return amp * np.exp(- (x - mean)**2 / (2 * sigma**2))


def generate_gaussian_cube(shape=(100, 25, 25), sigma=8., amp=1.,
                           noise=None, spec_scale=1 * u.km / u.s,
                           pixel_scale=1 * u.arcsec,
                           beamfwhm=3 * u.arcsec,
                           v0=None,
                           vel_surface=None,
                           seed=247825498,
                           use_dask=None):
    '''
    Generate a SpectralCube with Gaussian profiles.

    The peak velocity positions can be given with `vel_surface`. Otherwise,
    the peaks of the profiles are randomly assigned positions in the cubes.
    This is primarily to test shuffling and stacking of spectra, rather than
    trying to be being physically meaningful.

    Returns
    -------
    spec_cube : SpectralCube
        The generated cube.
    mean_positions : array
        The peak positions in the cube.
    '''

    test_cube = np.empty(shape)
    mean_positions = np.empty(shape[1:])

    spec_middle = int(shape[0] / 2)
    spec_quarter = int(shape[0] / 4)

    if v0 is None:
        v0 = 0

    with NumpyRNGContext(seed):

        spec_inds = np.mgrid[-spec_middle:spec_middle] * spec_scale.value
        if len(spec_inds) == 0:
            spec_inds = np.array([0])
        spat_inds = np.indices(shape[1:])
        for y, x in zip(spat_inds[0].flatten(), spat_inds[1].flatten()):
            # Lock the mean to within 25% from the centre
            if vel_surface is not None:
                mean_pos = vel_surface[y,x]
            else:
                mean_pos = \
                    np.random.uniform(low=spec_inds[spec_quarter],
                                      high=spec_inds[spec_quarter + spec_middle])
            test_cube[:, y, x] = gaussian(spec_inds, amp, mean_pos, sigma)
            mean_positions[y, x] = mean_pos + v0
            if noise is not None:
                test_cube[:, y, x] += np.random.normal(0, noise, shape[0])

    test_hdu = generate_hdu(test_cube, pixel_scale, spec_scale, beamfwhm,
                            spec_inds[0] + v0)

    spec_cube = SpectralCube.read(test_hdu, use_dask=use_dask)

    mean_positions = mean_positions * spec_scale.unit

    return spec_cube, mean_positions
