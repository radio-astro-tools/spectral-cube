import numpy as np

from astropy import units as u

def stack_spectra(cube, velocity_field, average=np.nanmean):
    """
    Stack all spectra in a cube by shifting each spectrum in velocity and then
    averaging

    Parameters
    ----------
    cube : SpectralCube
        The cube
    velocity_field : Quantity
        A Quantity array with m/s or equivalent units
    average : function
        A function that can operate over a list of numpy arrays (and accepts
        ``axis=0``) to average the spectra.  `numpy.nanmean` is the default,
        though one might consider `numpy.mean` or `numpy.median` as other
        options.

    """

    if not all(cube.shape[1:] == velocity_field.shape):
        raise ValueError("Cube spatial dimensions must match velocity field map dimensions")

    if not cube.spectral_axis.unit.is_equivalent(u.m/u.s):
        # select the mean velocity/wavelength as the reference
        ref = cube.spectral_axis.mean()
        vcube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=ref)
    else:
        vcube = cube

    velax = vcube.spectral_axis

    spectra = []

    for index, velocity in np.ndenumerate(velocity_field):
        if np.isfinite(velocity):
            shifted_vel = velax - velocity
            newspec = np.interpolate(shifted_vel, velax, vcube[:, index[0], index[1]])
            spectra.append(newspec)

    meanspec = average(spectra, axis=0)

    template_hdu = cube.with_spectral_unit(u.GHz)[:,0,0].hdu

    template_hdu.data = meanspec

    return template_hdu

def stack_cube(cube, linelist, vmin, vmax, average=np.nanmean):
    """
    Create a stacked cube by averaging on a common velocity grid.

    Parameters
    ----------
    cube : SpectralCube
        The cube
    linelist : list of Quantities
        An iterable of Quantities representing line rest frequencies
    vmin / vmax : Quantity
        Velocity-equivalent quantities specifying the velocity range to average
        over
    average : function
        A function that can operate over a list of numpy arrays (and accepts
        ``axis=0``) to average the spectra.  `numpy.nanmean` is the default,
        though one might consider `numpy.mean` or `numpy.median` as other
        options.
    """
    
    line_cube = cube.with_spectral_unit(u.km/u.s,
                                        velocity_convention='radio',
                                        rest_value=linelist[0])
    reference_cube = line_cube.spectral_slab(vmin, vmax)

    cutout_cubes = [reference_cube[:].value]

    for restval in linelist[1:]:
        line_cube = cube.with_spectral_unit(u.km/u.s,
                                            velocity_convention='radio',
                                            rest_value=restval)
        line_cutout = line_cube.spectral_slab(vmin, vmax)

        regridded = line_cutout.spectral_interpolate(reference_cube.spectral_axis)

        cutout_cubes.append(regridded[:].value)

    stacked_cube = average(cutout_cubes, axis=0)

    hdu = reference_cube.hdu

    hdu.data = stacked_cube

    return hdu
