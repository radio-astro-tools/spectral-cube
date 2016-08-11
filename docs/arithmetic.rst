Spectral Cube Arithmetic
========================

Simple arithmetic operations between cubes and scalars, broadcastable numpy
arrays, and other cubes are possible.  However, such operations should be
performed with caution because they require loading the whole cube into memory
and will generally create a new cube in memory.

Examples::

    >>> import astropy.units as u
    >>> from spectral_cube import SpectralCube
    >>> cube = SpectralCube.read('adv.fits')  # doctest: +SKIP
    >>> cube2 = cube * 2  # doctest: +SKIP
    >>> cube3 = cube + 1.5*u.K # doctest: +SKIP
    >>> cube4 = cube2 + cube3 # doctest: +SKIP

Each of these cubes is a new cube in memory.  Note that for addition and
subtraction, the units must be equivalent to those of the cube.

Please see :ref:`doc_handling_large_datasets` for details on how to perform
arithmetic operations on a small subset of data at a time.
