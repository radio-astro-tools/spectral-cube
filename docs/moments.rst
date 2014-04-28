Moment maps and statistics
==========================

Producing moment maps from a
:class:`~spectral_cube.SpectralCube` instance is
straightforward::

    >>> moment_0 = cube.moment(order=0)
    >>> moment_1 = cube.moment(order=1)
    >>> moment_2 = cube.moment(order=2)

By default, moments are computed along the spectral dimension, but it is also
possible to pass the ``axis`` argument to compute them along a different
axis::

    >>> moment_0_along_x = cube.moment(order=0, axis=2)

The moment maps returned are :class:`~spectral_cube.MomentMap` instances,
which act like :class:`~astropy.units.Quantity` objects, and also have
convenience methods for writing to a file::

    >>> moment_0.write('moment0.fits')

 and converting the data and WCS to a FITS HDU::

    >>> moment_0.hdu
    <astropy.io.fits.hdu.image.PrimaryHDU at 0x10d6ec510>

The conversion to HDU objects makes it very easy to use the moment map with
plotting packages such as APLpy::

    >>> import aplpy
    >>> import aplpy
    >>> f = aplpy.FITSFigure(moment_0.hdu)
    >>> f.show_colorscale()
    >>> f.save('moment_0.png')
