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
          the moments, please see :class:`~spectral_cube.SpectralCube.moment`.
          For linewidth maps, see the `Linewidth maps`_ section.
          
You may also want to convert the unit of the datacube into a velocity one before
you can obtain a genuine velocity map via a 1st moment map. So first it will be necessary to 
apply the :class:`~spectral_cube.SpectralCube.with_spectral_unit` method from this package with the proper attribute settings::

    >>> nii_cube = cube.with_spectral_unit(u.km/u.s,
                                           velocity_convention='optical',
                                           rest_value=6584*u.AA)  # doctest: +SKIP

Note that the ``rest_value`` in the above code refers to the wavelength of the targeted line 
in the 1D spectrum corresponding to the 3rd dimension. Also, since not all velocity values are relevant, 
next we will use the :class:`~spectral_cube.SpectralCube.spectral_slab` method to slice out the chunk of 
the cube that actually contains the line::

    >>> nii_cube = cube.with_spectral_unit(u.km/u.s,
                                           velocity_convention='optical',
                                           rest_value=6584*u.AA)  # doctest: +SKIP
    >>> nii_subcube = nii_cube.spectral_slab(-60*u.km/u.s,-20*u.km/u.s)  # doctest: +SKIP
    
Finally, we can now generate the 1st moment map containing the expected velocity structure::

    >>> moment_1 = nii_subcube.moment(order=1)  # doctest: +SKIP

The moment maps returned are :class:`~spectral_cube.lower_dimensional_structures.Projection` instances,
which act like :class:`~astropy.units.Quantity` objects, and also have
convenience methods for writing to a file::

    >>> moment_0.write('moment0.fits')  # doctest: +SKIP
    >>> moment_1.write('moment1.fits')  # doctest: +SKIP

and converting the data and WCS to a FITS HDU::

    >>> moment_0.hdu  # doctest: +SKIP
    <astropy.io.fits.hdu.image.PrimaryHDU at 0x10d6ec510>

The conversion to HDU objects makes it very easy to use the moment map with
plotting packages such as `aplpy <https://aplpy.github.io/>`_::

    >>> import aplpy  # doctest: +SKIP
    >>> f = aplpy.FITSFigure(moment_0.hdu)  # doctest: +SKIP
    >>> f.show_colorscale()  # doctest: +SKIP
    >>> f.save('moment_0.png')  # doctest: +SKIP

There is a shortcut for the above, if you have aplpy_ installed::

    >>> moment_0.quicklook('moment_0.png')

will create the quicklook grayscale image and save it to a png all in one go.

Moment map equations
^^^^^^^^^^^^^^^^^^^^
 
The moments are defined below, using :math:`v` for the spectral (velocity,
frequency, wavelength, or energy) axis and :math:`I_v` as the intensity,
or otherwise measured flux, value in a given spectral channel.

The equation for the 0th moment is:

.. math:: M_0 = \int I_v  dv

The equation for the 1st moment is:

.. math:: M_1 = \frac{\int v I_v  dv}{\int I_v dv} = \frac{\int v I_v dv}{M_0}
   
Higher-order moments (:math:`N\geq2`) are defined as:

.. math:: M_N = \frac{\int I_v (v - M_1)^N dv}{M_0}


Descriptions for the three most common moments used are:

* 0th moment - the integrated intensity over the spectral line.  Units are cube
  unit times spectral axis unit (e.g., K km/s).
* 1st moment - the the intensity-weighted velocity of the spectral line.  The
  unit is the same as the spectral axis unit (e.g., km/s)
* 2nd moment - the velocity dispersion or the width of the spectral line.  The
  unit is the spectral axis unit squared (e.g., :math:`km^2/s^2`).  To obtain measurements
  of the linewidth in spectral axis units, see `Linewidth maps`_ below


Linewidth maps
--------------

Line width maps based on the 2nd moment maps, as defined above, can be made
with either of these two commands::

    >>> sigma_map = cube.linewidth_sigma()  # doctest: +SKIP
    >>> fwhm_map = cube.linewidth_fwhm()  # doctest: +SKIP

``~spectral_cube.SpectralCube.linewidth_sigma`` computes a sigma linewidth map
along the spectral axis, where sigma is the width of a Gaussian, while 
``~spectral_cube.SpectralCube.linewidth_fwhm`` computes a FWHM
linewidth map along the same spectral axis.

The linewidth maps are related to the second moment by

.. math:: \sigma = \sqrt{M_2} \\
          FWHM = \sigma \sqrt{8 ln{2}} 

These functions return :class:`~spectral_cube.lower_dimensional_structures.Projection` instances as for the
`Moment maps`_.
