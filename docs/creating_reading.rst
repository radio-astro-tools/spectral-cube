Creating and/or reading a spectral cube
=======================================

Importing
---------

The :class:`~spectral_cube.SpectralCube` class is used to
represent 3-dimensional datasets (two positional dimensions and one spectral
dimension) with a World Coordinate System (WCS) projection that describes the
mapping from pixel to world coordinates and vice-versa. The class is imported
with::

    >>> from spectral_cube import SpectralCube

Reading from a file
-------------------

In most cases, you are likely to read in an existing spectral cube from a
file. The reader in ``spectral_cube`` is designed to be able to deal with any
arbitrary axis order and always return a consistently oriented spectral cube
(see :doc:`accessing`). To read in a file, use the
:meth:`~spectral_cube.SpectralCube.read` method as follows::

     >>> cube = SpectralCube.read('L1448_13CO.fits')

This will always read the Stokes I parameter in the file. For information on
accessing other Stokes parameters, see :doc:`stokes`.

.. note:: In most cases, the FITS reader should be able to open the file in
          *memory-mapped* mode, which means that the data is not immediately
          read, but is instead read as needed when data is accessed. This
          allows large files (including larger than memory) to be accessed.
          However, note that certain FITS files cannot be opened in
          memory-mapped mode, in particular compressed (e.g. ``.fits.gz``)
          files. See the :doc:`big_data` page for more details about dealing
          with large data sets.

Direct Initialization
---------------------

If you are interested in directly creating a
:class:`~spectral_cube.SpectralCube` instance, you can do so using a 3-d
Numpy-like array with a 3-d :class:`~astropy.wcs.WCS` object::

    >>> cube = SpectralCube(data=data, wcs=wcs)

Here ``data`` can be any Numpy-like array, including *memory-mapped* Numpy
arrays (as mentioned in `Reading from a file`_, memory-mapping is a technique
that avoids reading the whole file into memory and instead accessing it from
the disk as needed).


