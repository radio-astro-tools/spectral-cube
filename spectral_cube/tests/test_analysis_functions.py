
import pytest
import warnings
import numpy as np
import astropy.units as u
# from astropy.modeling import models, fitting

from ..analysis_utilities import stack_spectra, fourier_shift, stack_cube
from .utilities import generate_gaussian_cube, gaussian
from ..utils import BadVelocitiesWarning

def test_shift():

    amp = 1
    v0 = 0 * u.m / u.s
    sigma = 8
    spectral_axis = np.arange(-50, 51) * u.m / u.s

    true_spectrum = gaussian(spectral_axis.value,
                             amp, v0.value, sigma)

    # Shift is an integer, so rolling is equivalent
    rolled_spectrum = np.roll(true_spectrum, 10)

    shift_spectrum = fourier_shift(true_spectrum, 10)

    np.testing.assert_allclose(shift_spectrum,
                               rolled_spectrum,
                               rtol=1e-4)

    # With part masked
    masked_spectrum = true_spectrum.copy()
    mask = np.abs(spectral_axis.value) <= 30
    masked_spectrum[~mask] = np.NaN

    rolled_mask = np.roll(mask, 10)
    rolled_masked_spectrum = rolled_spectrum.copy()
    rolled_masked_spectrum[~rolled_mask] = np.NaN

    shift_spectrum = fourier_shift(masked_spectrum, 10)

    np.testing.assert_allclose(shift_spectrum,
                               rolled_masked_spectrum,
                               rtol=1e-4)


def test_stacking(use_dask):
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.
    '''

    amp = 1.
    v0 = 0. * u.km / u.s
    sigma = 8.
    noise = None
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise,
                               shape=shape, use_dask=use_dask)

    true_spectrum = gaussian(test_cube.spectral_axis.value,
                             amp, v0.value, sigma)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False, pad_edges=False)

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


def test_cube_stacking(use_dask):
    '''
    Test passing a list of cubes

    This test simply averages two copies of the same thing.

    A more thorough test might be to verify that cubes with different frequency
    supports also yield good results.
    '''

    amp = 1.
    sigma = 8.
    noise = None
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise,
                               shape=shape, use_dask=use_dask)

    test_cube1 = test_cube.with_spectral_unit(u.GHz, rest_value=1*u.GHz, velocity_convention='radio')
    test_cube2 = test_cube.with_spectral_unit(u.GHz, rest_value=2*u.GHz, velocity_convention='radio')

    vmin = -10*u.km/u.s
    vmax = 10*u.km/u.s

    # Stack two cubes
    stacked = stack_cube([test_cube1, test_cube2], linelist=[1.,2.]*u.GHz,
                         vmin=vmin, vmax=vmax, average=np.nanmean,
                         convolve_beam=None, return_cutouts=False)

    np.testing.assert_allclose(stacked.filled_data[:],
                               test_cube.spectral_slab(vmin, vmax).filled_data[:])

    # Stack one cube with two frequencies, one that's out of band
    stacked = stack_cube(test_cube1, linelist=[1.,2.]*u.GHz,
                         vmin=vmin, vmax=vmax, average=np.nanmean,
                         convolve_beam=None, return_cutouts=False)

    np.testing.assert_allclose(stacked.filled_data[:],
                               test_cube.spectral_slab(vmin, vmax).filled_data[:])

    # TODO: add tests of multiple lines in the same cube
    # (this requires a different test cube setup)



def test_stacking_badvels(use_dask):
    '''
    Regression test for #493: don't include bad velocities when stacking
    '''

    amp = 1.
    v0 = 0. * u.km / u.s
    sigma = 8.
    noise = None
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise,
                               shape=shape, use_dask=use_dask)

    true_spectrum = gaussian(test_cube.spectral_axis.value,
                             amp, v0.value, sigma)

    test_vels[12,11] = 500*u.km/u.s

    with pytest.warns(BadVelocitiesWarning,
                      match='Some velocities are outside the allowed range and will be'):
        # Stack the spectra in the cube
        stacked = \
            stack_spectra(test_cube, test_vels, v0=v0,
                          stack_function=np.nanmean,
                          xy_posns=None, num_cores=1,
                          chunk_size=-1,
                          progressbar=False, pad_edges=False)

    # Calculate residuals (the one bad value shouldn't have caused a problem)
    resid = np.abs(stacked.value - true_spectrum)
    assert np.std(resid) <= 1e-3


def test_stacking_reversed_specaxis(use_dask):
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.
    '''

    amp = 1.
    v0 = 0. * u.km / u.s
    sigma = 8.
    noise = None
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise,
                               shape=shape, spec_scale=-1. * u.km / u.s, use_dask=use_dask)

    true_spectrum = gaussian(test_cube.spectral_axis.value,
                             amp, v0.value, sigma)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False, pad_edges=False)

    # Calculate residuals
    resid = np.abs(stacked.value - true_spectrum)
    assert np.std(resid) <= 1e-3

    # The stacked spectrum should have the same spectral axis
    np.testing.assert_allclose(stacked.spectral_axis.value,
                               test_cube.spectral_axis.value)


def test_stacking_wpadding(use_dask):
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.
    '''

    amp = 1.
    sigma = 8.
    v0 = 0. * u.km / u.s
    noise = None
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(shape=shape, amp=amp, sigma=sigma, noise=noise, use_dask=use_dask)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False, pad_edges=True)

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


def test_padding_direction(use_dask):

    amp = 1.
    sigma = 8.
    v0 = 0. * u.km / u.s
    noise = None
    shape = (100, 2, 2)

    vel_surface = np.array([[0, 5], [5, 10]])

    test_cube, test_vels = \
        generate_gaussian_cube(shape=shape, amp=amp, sigma=sigma, noise=noise,
                               vel_surface=vel_surface, use_dask=use_dask)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False, pad_edges=True)

    true_spectrum = gaussian(stacked.spectral_axis.value,
                             amp, v0.value, sigma)

    # now check that the stacked spectral axis is right
    # (all shifts are negative, so vmin < -50 km/s, should be -60?)
    assert stacked.spectral_axis.min() == -60*u.km/u.s
    assert stacked.spectral_axis.max() == 49*u.km/u.s

    # Calculate residuals
    resid = np.abs(stacked.value - true_spectrum)
    assert np.std(resid) <= 1e-3


def test_stacking_woffset(use_dask):
    '''
    Use a set of identical Gaussian profiles randomly offset to ensure the
    shifted spectrum has the correct properties.

    Make sure the operations aren't affected by absolute velocity offsets

    '''

    amp = 1.
    sigma = 8.
    v0 = 100. * u.km / u.s
    noise = None
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(shape=shape, amp=amp, sigma=sigma, noise=noise,
                               v0=v0.value, use_dask=use_dask)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False, pad_edges=True)

    true_spectrum = gaussian(stacked.spectral_axis.value,
                             amp, v0.value, sigma)

    # Calculate residuals
    resid = np.abs(stacked.value - true_spectrum)
    assert np.std(resid) <= 1e-3

    # The spectral axis should be padded by ~25% on each side
    stack_shape = int(test_cube.shape[0] * 1.5)
    # This is rounded, so the shape could be +/- 1
    assert (stacked.size == stack_shape) or (stacked.size == stack_shape - 1) \
        or (stacked.size == stack_shape + 1)


def test_stacking_shape_failure(use_dask):
    """
    Regression test for #466
    """
    amp = 1.
    v0 = 0. * u.km / u.s
    sigma = 8.
    noise = None
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise,
                               shape=shape, use_dask=use_dask)

    # make the test_vels array the wrong shape
    test_vels = test_vels[:-1, :-1]

    with pytest.raises(ValueError) as exc:
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False, pad_edges=False)

    assert 'Velocity surface map does not match' in exc.value.args[0]


    test_vels = np.ones(shape[1:], dtype='float') + np.nan

    with pytest.raises(ValueError) as exc:
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False, pad_edges=False)

    assert "velocity_surface contains no finite values" in exc.value.args[0]


def test_stacking_noisy(use_dask):

    # Test stack w/ S/N of 0.2
    # This is cheating b/c we know the correct peak velocities, but serves as
    # a good test that the stacking is working.

    amp = 1.
    sigma = 8.
    v0 = 0 * u.km / u.s
    noise = 5.0
    shape = (100, 25, 25)

    test_cube, test_vels = \
        generate_gaussian_cube(amp=amp, sigma=sigma, noise=noise,
                               shape=shape, use_dask=use_dask)

    # Stack the spectra in the cube
    stacked = \
        stack_spectra(test_cube, test_vels, v0=v0,
                      stack_function=np.nanmean,
                      xy_posns=None, num_cores=1,
                      chunk_size=-1,
                      progressbar=False,
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
