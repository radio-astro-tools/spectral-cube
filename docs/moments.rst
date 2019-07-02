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

    >>> nii_cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=6584*u.AA)  # doctest: +SKIP

Note that the ``rest_value`` in the above code refers to the wavelength of the targeted line 
in the 1D spectrum corresponding to the 3rd dimension. Also, since not all velocity values are relevant, 
next we will use the :class:`~spectral_cube.SpectralCube.spectral_slab` method to slice out the chunk of 
the cube that actually contains the line::

    >>> nii_cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='optical', rest_value=6584*u.AA)  # doctest: +SKIP
    >>> nii_subcube = nii_cube.spectral_slab(-60*u.km/u.s,-20*u.km/u.s)  # doctest: +SKIP
    
Finally, we can now generate the 1st moment map containing the expected velocity structure::

    >>> moment_1 = nii_subcube.moment(order=1)  # doctest: +SKIP

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

The equation for the N-th moment, where N is usually an integer between -1 and some arbitrary
positive integer say 11, is as follows:

.. math:: M_N = \frac{\int I (l - M_l)^N dl}{M_0}

Descriptions for the twelve most common moments used are as follows:

moments=-1 - the mean value of the spectrum  
moments=0  - the integrated value of the spectrum, can be used for computing line ratios
moments=1  - the intensity weighted coordinate; by convention used to get the ’velocity fields’
moments=2  - the intensity weighted dispersion of the coordinate; by convention used to get the ’velocity dispersion’  
moments=3  - the median of the cube I
moments=4  - the median coordinate  
moments=5  - the standard deviation (SD) about the mean of the spectrum  
moments=6  - the root mean square of the spectrum  
moments=7  - the absolute mean deviation of the spectrum  
moments=8  - the maximum value of the spectrum  
moments=9  - the coordinate of the maximum value of the spectrum  
moments=10 - the minimum value of the spectrum  
moments=11 - the coordinate of the minimum value of the spectrum


Linewidth maps
--------------

Making linewidth maps (sometimes referred to as second moment maps in radio
astronomy), you can use::

    >>> sigma_map = cube.linewidth_sigma()  # doctest: +SKIP
    >>> fwhm_map = cube.linewidth_fwhm()  # doctest: +SKIP

These also return :class:`~spectral_cube.lower_dimensional_structures.Projection` instances as for the
`Moment maps`_.
