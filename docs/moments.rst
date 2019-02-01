Moment maps and statistics
==========================

Moment maps
-----------

Producing moment maps from a
:class:`~spectral_cube.SpectralCube` instance is
straightforward::

    >>> moment_0 = cube.moment(order=0)  # doctest: +SKIP
    >>> moment_1 = cube.moment(order=1)  # doctest: +SKIP
    >>> moment_2 = cube.moment(order=2)  # doctest: +SKIP

By default, moments are computed along the spectral dimension, but it is also
possible to pass the ``axis`` argument to compute them along a different
axis::

    >>> moment_0_along_x = cube.moment(order=0, axis=2)  # doctest: +SKIP

.. note:: These follow the mathematical definition of moments, so the second
          moment is computed as the variance. For the actual formulas used for
          the moments, please see `the relevant documentation here 
          <https://spectral-cube.readthedocs.io/en/latest/api/spectral_cube.SpectralCube.html#spectral_cube.SpectralCube.moment>`_.
          For linewidth maps, see the
          `Linewidth maps`_ section.
          
You may also want to convert the unit of the datacube into a velocity one before
you can obtain a genuine velocity map via a 1st moment map. So first it will be necessary to 
apply the :class:`~spectral_cube.SpectralCube.with_spectral_unit` method from this package with the proper attribute settings::

    >>> nii_cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=6584*u.AA)

Note that the `rest_value` in the above code refers to the wavelength of the targeted line 
in the 1D spectrum corresponding to the 3rd dimension. Also, since not all velocity values are relevant, 
next we will use the :class:`~spectral_cube.SpectralCube.spectral_slab` method to slice out the chunk of 
the cube that actually contains the line::

    >>> nii_cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=6584*u.AA).spectral_slab(-60*u.km/u.s,-20*u.km/u.s)
    
Finally, we can now generate the 1st moment map containing the expected velocity structure::

    >>> moment_1 = nii_cube_2.moment(order=1)  # doctest: +SKIP

The moment maps returned are :class:`~spectral_cube.lower_dimensional_structures.Projection` instances,
which act like :class:`~astropy.units.Quantity` objects, and also have
convenience methods for writing to a file::

    >>> moment_0.write('moment0.fits')  # doctest: +SKIP
    >>> momemt_1.write('moment1.fits')  # doctest: +SKIP

and converting the data and WCS to a FITS HDU::

    >>> moment_0.hdu  # doctest: +SKIP
    <astropy.io.fits.hdu.image.PrimaryHDU at 0x10d6ec510>

The conversion to HDU objects makes it very easy to use the moment map with
plotting packages such as APLpy::

    >>> import aplpy  # doctest: +SKIP
    >>> f = aplpy.FITSFigure(moment_0.hdu)  # doctest: +SKIP
    >>> f.show_colorscale()  # doctest: +SKIP
    >>> f.save('moment_0.png')  # doctest: +SKIP

Linewidth maps
--------------

Making linewidth maps (sometimes referred to as second moment maps in radio
astronomy), you can use::

    >>> sigma_map = cube.linewidth_sigma()  # doctest: +SKIP
    >>> fwhm_map = cube.linewidth_fwhm()  # doctest: +SKIP

These also return :class:`~spectral_cube.lower_dimensional_structures.Projection` instances as for the
`Moment maps`_.
