Selecting subsets
=================

Extracting a spectral slab
--------------------------

Given a spectral cube, it is easy to extract a sub-cube covering only a subset
of the original range in the spectral axis. To do this, you can use the
:meth:`~spectral_cube.SpectralCube.spectral_slab` method. This
method takes lower and upper bounds for the spectral axis, as well as an
optional rest frequency, and returns a new
:class:`~spectral_cube.SpectralCube` instance. The bounds can
be specified as a frequency, wavelength, or a velocity relative to a rest
frequency. If the latter, then the rest frequency needs to be specified. The
bounds and the rest frequency (if applicable) should be given as Astropy
:class:`Quantities <astropy.units.Quantity>` as follows:

    >>> from astropy import units as u
    >>> co_1_0 = cube.spectral_slab(-50 * u.km / u.s, +50 * u.km / u.s,
                                    rest_frequency=115.27120 * u.GHz)

Regardless of what units the original cube was in, the
:meth:`~spectral_cube.SpectralCube.spectral_slab` can determine
how to convert the velocities to frequencies if needed. The resulting cube
``co_1_0`` (which is also a
:class:`~spectral_cube.SpectralCube` instance) then contains
all channels that overlap with the range -50 to 50 km/s relative to the 12CO
1-0 line.

Extracting a sub-cube by indexing
---------------------------------

It is also easy to extract a sub-cube from pixel coordinates using standard
Numpy slicing notation::

    >>> sub_cube = cube[:100, 10:50, 10:50]

This returns a new :class:`~spectral_cube.SpectralCube` object
with updated WCS information.