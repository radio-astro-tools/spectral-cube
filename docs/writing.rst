Writing spectral cubes
======================

You can write out a :class:`~spectral_cube.SpectralCube`
instance by making use of the
:meth:`~spectral_cube.SpectralCube.write` method::

    >>> cube.write('new_cube.fits', format='fits')  # doctest: +SKIP

