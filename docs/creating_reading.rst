Creating and/or reading a spectral cube
=======================================

Initializing
------------

The :class:`~spectral_cube.SpectralCube` class is used to
represent 3-dimensional datasets (two positional dimensions and one spectral
dimension) with a World Coordinate System (WCS) projection that describes the
mapping from pixel to world coordinates and vice-versa. The class is imported
with::

    >>> from spectral_cube import SpectralCube

The easiest way to instantiate a
:class:`~spectral_cube.SpectralCube` is to pass a 3-d numpy
array along with a 3-d :class:`~astropy.wcs.WCS` object::

    >>> cube = SpectralCube(data=data, wcs=wcs)

Here ``data`` can be a normal Numpy array, or a *memory-mapped* Numpy array
(memory-mapping is a technique that avoids reading the whole file into memory
and instead accessing it from the disk as needed). When reading in data from
FITS files with `astropy.io.fits <LINK>`_, the ``memmap`` keyword argument
allows you to access the data via a memory-mapped array.

Reading
-------

In practice, you will most often want to read a spectral cube from a file. The
reader in ``spectral_cube`` is designed to be able to deal with any arbitrary
axis order and always return a consistently oriented
spectral cube (see `Accessing data`_ below). To read in a file, use the
:meth:`~spectral_cube.SpectralCube.read` class method as
follows::

     >>> cube = SpectralCube.read('L1448_13CO.fits')

This will always read the Stokes I parameter in the file. For information on
accessing other Stokes parameters, see the `Stokes parameters`_ section.

.. note:: In most cases, the FITS reader should be able to open the file in
          memory-mapped mode, which means that the data is not immediately
          read, but is instead read as needed when data is accessed. This
          allows large files (including larger than memory) to be accessed.
          However, note that certain FITS files cannot be opened in
          memory-mapped mode, in particular compressed (e.g. ``.fits.gz``)
          files.
