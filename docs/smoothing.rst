Smoothing
---------

There are two types of smoothing routine available in ``spectral_cube``:
spatial and spectral.

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

Common Beam selection
^^^^^^^^^^^^^^^^^^^^^
You may want to convolve your cube to the smallest beam that is still larger
than all contained beams.  To do this, you can use the
`~radio_beam.Beams.common_beam` tool.  For example::

    common_beam = cube.beams.common_beam()
    new_cube = cube.convolve_to(common_beam)

Sometimes, you'll encounter the error "Could not find common beam to deconvolve
all beams." This is a real issue, as the algorithms we have in hand so far do
not always converge on a common containing beam. There are two ways to get the
algorithm to converge to a valid common beam:

1. **Changing the tolerance.** - You can try to change the tolerance used in the
`~radio_beam.commonbeam.getMinVolEllipse` code by
passing ``tolerance=1e-5`` to the common beam function::

    cube.beams.common_beam(tolerance=1e-5)

Convergence may be met by either increasing or decreasing the tolerance; it
depends on having the algorithm not step within the minimum enclosing ellipse,
leading to the error. Note that decreasing the tolerance by an order of magnitude
will require an order of magnitude more iterations for the algorithm to converge
and will take longer to run.

2. **Changing epsilon** - A second parameter ``epsilon`` controls the fraction
to overestimate the beam size, ensuring that solutions that are marginally
smaller than the common beam will not be found by the algorithm::

    cube.beams.common_beam(epsilon=1e-3)

The default value of ``epsilon=1e-3`` will sample points 0.1% larger than the
edge of each beam in the set. Increasing ``epsilon`` ensures that a valid common
beam can be found, avoiding the tolerance issue, but will result in
overestimating the common beam area. For most radio data sets, where the beam
is oversampled by :math:`\sim 3 \mbox{-5}` pixels, moderate increases in
``epsilon`` will increase the common beam area far less than a pixel area, making
the overestimation negligible.

We recommend testing different values of tolerance to find convergence, and if
the error persists, to then slowly increase epsilon until a valid common beam is
found. More information can be found in the
`radio-beam documentation <https://radio-beam.readthedocs.io/en/latest/>`_.

Alternative approach to spatial smoothing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There is an alternative way to spatially smooth a data cube, which is using the
:meth:`~spectral_cube.SpectralCube.spatial_smooth` method. This is an example
of how you can do this::

    from spectral_cube import SpectralCube
    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel

    cube = SpectralCube.read('/some_path/some_file.fits')
    kernel = Gaussian2DKernel(x_stddev=1)
    new_cube = cube.spatial_smooth(kernel)
    new_cube.write('/some_path/some_other_file.fits')

``x_stddev`` specifies the width of the `Gaussian kernel <http://docs.astropy.org/en/stable/api/astropy.convolution.Gaussian2DKernel.html>`_.
Any `astropy convolution <kernel http://docs.astropy.org/en/stable/convolution/kernels.html>`_
is acceptable.

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
(0.25^2 - 0.1^2)^0.5 = 0.229 km/s. For simplicity, it can be
done in the unit of pixel.  In our example, each channel is 0.1 km/s wide::

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
    kernel = Gaussian1DKernel(gaussian_width.value)
    new_cube = cube.spectral_smooth(kernel)
    new_cube.write('newfile.fits')

`gaussian_width` is in pixel units but is defined as a unitless `~astropy.units.Quantity`.
By using `gaussian_width.value`, we convert the pixel width into a float.

Reprojection
============

Smoothing changes the data properties but not the underlying grid.  It is often
helpful to re-project the data in either the spatial or spectral directions to
match cubes in pixel space.

Spatial reprojection can be achieved with the `SpectralCube.reproject` method::

    reproj_cube = cube.reproject(new_header)

This method will loop over each channel using the `reproject
<https://reproject.readthedocs.io/en/stable/>`_ model to regrid each channel.
The regridding is done with interpolation.

It may also be necessary to first spectrally regrid, which can be done
following the `Spectral Smoothing`_ approach above.

For a full example showing appropriate resampling and reprojection to match two
cubes, please see the `reprojection tutorial
<https://github.com/radio-astro-tools/tutorials/pull/17>`_.


