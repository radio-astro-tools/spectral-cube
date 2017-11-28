
import numpy as np
import astropy.units as u
from scipy.optimize import curve_fit

from ..analysis_utilities import stack_spectra
from .utilities import generate_gaussian_cube, gaussian


def test_stacking():
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.
    '''

    amp = 1.
    sigma = 8.
    noise = None

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=0 * u.km / u.s,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      verbose=False, pad_edges=False)

    # Now fit a Gaussian to the mean stacked profile.

    fit_vals = curve_fit(gaussian, stacked.spectral_axis.value,
                         stacked.value)[0]

    np.testing.assert_allclose(fit_vals, np.array([amp, 0.0, sigma]),
                               atol=1e-3)

    # The stacked spectrum should have the same spectral axis
    np.testing.assert_allclose(stacked.spectral_axis.value,
                               test_cube.spectral_axis.value)


def test_stacking_wpadding():
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.
    '''

    amp = 1.
    sigma = 8.
    noise = None
    shape = (100, 100, 100)

    test_cube, test_vels = \
        generate_gaussian_cube(shape=shape, amp=amp, sigma=sigma, noise=noise)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=0 * u.km / u.s,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      verbose=False, pad_edges=True)

    # Now fit a Gaussian to the mean stacked profile.

    fit_vals = curve_fit(gaussian, stacked.spectral_axis.value,
                         stacked.value)[0]

    np.testing.assert_allclose(fit_vals, np.array([amp, 0.0, sigma]),
                               atol=1e-3)

    # The spectral axis should be padded by ~25% on each side
    stack_shape = int(test_cube.shape[0] * 1.5)
    # This is rounded, so the shape could be +/- 1
    assert (stacked.size == stack_shape) or (stacked.size == stack_shape - 1) \
        or (stacked.size == stack_shape + 1)


def test_stacking_noisy():

    # Test stack w/ S/N of 0.2
    # This is cheating b/c we know the correct peak velocities, but serves as
    # a good test that the stacking is working.

    amp = 1.
    sigma = 8.
    v0 = 0 * u.km / u.s
    noise = 5.0

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      verbose=False,
                      pad_edges=True)

    stacked.quicklook()

    # Now fit a Gaussian to the mean stacked profile.

    fit_vals, fit_cov = curve_fit(gaussian, stacked.spectral_axis.value,
                                  stacked.value)
    fit_errs = np.sqrt(np.diag(fit_cov))

    # Check that the fit is consistent with the true values within 1-sigma err
    for fit_val, fit_err, true_val in zip(fit_vals, fit_errs,
                                          [amp, v0.value, sigma]):
        np.testing.assert_allclose(fit_val, true_val,
                                   atol=fit_err)
