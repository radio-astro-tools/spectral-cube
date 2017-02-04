Smoothing
---------

There are two types of smoothing routine available in ``spectral_cube``:
spectral and spatial.

Spatial Smoothing
=================

The `~spectral_cube.SpectralCube.convolve_to` method will convolve each plane
of the cube to a common resolution, assuming the cube's resolution is known
in advanced and stored in the cube's ``beam`` or ``beams`` attribute.

A simple example::

    import radio_beam
    from spectral_cube import SpectralCube
    from astropy import units as u

    cube = SpectralCube.read('file.fits')
    beam = radio_beam.Beam(major=1*u.arcsec, minor=1*u.arcsec, pa=0*u.deg)
    new_cube = cube.convolve_to(beam)

Note that the :meth:`~spectral_cube.SpectralCube.convolve_to` method will work
for both :class:`~spectral_cube.VaryingResolutionSpectralCube` instances and
single-resolution :class:`~spectral_cube.SpectralCube` instances, but for a
:class:`~spectral_cube.VaryingResolutionSpectralCube`, the convolution kernel
will be different for each slice.

Spectral Smoothing
==================

Only :class:`~spectral_cube.SpectralCube` instances with a consistent beam can
be spectrally smoothed, so if you have a
:class:`~spectral_cube.VaryingResolutionSpectralCube`, you must convolve each
slice in it to a common resolution before spectrally smoothing.
:meth:`~spectral_cube.SpectralCube.spectral_smooth` will apply a convolution
kernel to each spectrum in turn. As of July 2016, a parallelized version is
partly written but incomplete.

Example::

    import radio_beam
    from spectral_cube import SpectralCube
    from astropy import units as u
    from astropy.convolution import Gaussian1DKernel

    cube = SpectralCube.read('file.fits')
    kernel = Gaussian1DKernel(2.5)
    new_cube = cube.spectral_smooth(kernel)

This can be useful if you want to interpolate onto a coarser grid but maintain
Nyquist sampling.  You can then use the
`~spectral_cube.SpectralCube.spectral_interpolate` method to regrid your
smoothed spectrum onto a new grid.

Say, for example, you have a cube with 0.5 km/s resolution, but you want to
resample it onto a 2 km/s grid.  You might then choose to smooth by a factor of
4, then downsample by the same factor::

    # cube.spectral_axis is np.arange(0,10,0.5) for this example
    new_axis = np.arange(0,10,2)*u.km/u.s
    fwhm_factor = np.sqrt(8*np.log(2))

    smcube = cube.spectral_smooth(Gaussian1DKernel(4/fwhm_factor))
    interp_Cube = smcube.spectral_interpolate(new_axis,
                                              suppress_smooth_warning=True)

We include the ``suppress_smooth_warning`` override because there is no way for
``SpectralCube`` to know if you've done the appropriate smoothing (i.e., making
sure that your new grid nyquist samples the data) prior to the interpolation
step.  If you don't specify this, it will still work, but you'll be warned that
you should preserve Nyquist sampling.

If you have a cube with 0.1 km/s resolution (where we assume resolution
corresponds to the fwhm of a gaussian), and you want to smooth it to 0.25 km/s
resolution, you can smooth the cube with a Gaussian Kernel that has a width of
(0.25<sup>2</sup> - 0.1<sup>2</sup>)<sup>0.5</sup> = 0.229 km/s. For
simplicity, it can be done in the unit of pixel.  In our example, each channel
is 0.1 km/s wide::

    import numpy as np
    from astropy import units as u
    from spectral_cube import SpectralCube
    from astropy.convolution import Gaussian1DKernel

    cube = SpectralCube.read('file.fits')
    fwhm_factor = np.sqrt(8*np.log(2))
    current_resolution = 0.1 * u.km/u.s
    target_resolution = 0.25 * u.km/u.s
    pixel_scale = 0.1 * u.km/u.s
    gaussian_width = ((target_resolution**2 - current_resolution**2)**0.5 /
                      pixel_scale / fwhm_factor)
    kernel = Gaussian1DKernel(gaussian_width)
    new_cube = cube.spectral_smooth(kernel)
    new_cube.write('newfile.fits')
