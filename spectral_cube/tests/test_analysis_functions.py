
import numpy as np
import astropy.units as u
# from astropy.modeling import models, fitting

from ..analysis_utilities import stack_spectra
from .utilities import generate_gaussian_cube, gaussian


def test_stacking():
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.
    '''

    amp = 1.
    v0 = 0. * u.km / u.s
    sigma = 8.
    noise = None

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise)

    true_spectrum = gaussian(test_cube.spectral_axis.value,
                             amp, v0.value, sigma)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      verbose=False, pad_edges=False)

    # Calculate residuals
    resid = np.abs(stacked.value - true_spectrum)
    assert np.std(resid) <= 1e-3

    # Now fit a Gaussian to the mean stacked profile.
    # fit_vals = fit_gaussian(stacked.spectral_axis.value, stacked.value)[0]

    # np.testing.assert_allclose(fit_vals, np.array([amp, v0.value, sigma]),
    #                            atol=1e-3)

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
    v0 = 0. * u.km / u.s
    noise = None
    shape = (100, 100, 100)

    test_cube, test_vels = \
        generate_gaussian_cube(shape=shape, amp=amp, sigma=sigma, noise=noise)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      verbose=False, pad_edges=True)

    true_spectrum = gaussian(stacked.spectral_axis.value,
                             amp, v0.value, sigma)

    # Calculate residuals
    resid = np.abs(stacked.value - true_spectrum)
    assert np.std(resid) <= 1e-3

    # Now fit a Gaussian to the mean stacked profile.
    # fit_vals = fit_gaussian(stacked.spectral_axis.value, stacked.value)[0]

    # np.testing.assert_allclose(fit_vals, np.array([amp, 0.0, sigma]),
    #                            atol=1e-3)

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
    shape = (100, 100, 100)

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise,
                               shape=shape)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      verbose=False,
                      pad_edges=True)

    true_spectrum = gaussian(stacked.spectral_axis.value,
                             amp, v0.value, sigma)

    # Calculate residuals
    resid = np.abs(stacked.value - true_spectrum)
    assert np.std(resid) <= noise / np.sqrt(shape[1] * shape[2])

    # Now fit a Gaussian to the mean stacked profile.
    # fit_vals, fit_errs = fit_gaussian(stacked.spectral_axis.value,
    #                                   stacked.value)

    # Check that the fit is consistent with the true values within 1-sigma err
    # for fit_val, fit_err, true_val in zip(fit_vals, fit_errs,
    #                                       [amp, v0.value, sigma]):
    #     np.testing.assert_allclose(fit_val, true_val,
    #                                atol=fit_err)


# def fit_gaussian(vels, data):
#     g_init = models.Gaussian1D()

#     fit_g = fitting.LevMarLSQFitter()

#     g_fit = fit_g(g_init, vels, data)

#     cov = fit_g.fit_info['param_cov']
#     if cov is None:
#         cov = np.zeros((3, 3)) * np.NaN
#     parvals = g_fit.parameters

#     parerrs = np.sqrt(np.diag(cov))

#     return parvals, parerrs
