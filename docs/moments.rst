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

.. note:: these follow the mathematical definition of moments, so the second
          moment is computed as the variance. For linewidth maps, see the
          `Linewidth maps`_ section.

The moment maps returned are :class:`~spectral_cube.Projection` instances,
which act like :class:`~astropy.units.Quantity` objects, and also have
convenience methods for writing to a file::

    >>> moment_0.write('moment0.fits')  # doctest: +SKIP

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

Making linewidth maps (sometimes referered to as second moment maps in radio
astronomy), you can use:

    >>> sigma_map = cube.linewidth_sigma()
    >>> fwhm_map = cube.linewidth_fwhm()

These also return :class:`~spectral_cube.Projection` instances as for the
`Moment maps`_.
