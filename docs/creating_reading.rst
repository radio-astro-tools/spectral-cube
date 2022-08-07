Creating/reading spectral cubes
===============================

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
file. The reader is designed to be able to deal with any
arbitrary axis order and always return a consistently oriented spectral cube
(see :doc:`accessing`). To read in a file, use the
:meth:`~spectral_cube.SpectralCube.read` method as follows::

     >>> cube = SpectralCube.read('L1448_13CO.fits')  # doctest: +SKIP

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


Reading images from file
------------------------

While spectral-cube is designed for cube analysis, in the course of normal
analysis procedures you are likely to need to load up one- and two-dimensional
subsets or views of the data.

You can load :class:`~spectral_cube.lower_dimensional_objects.Projection`
objects from 2D FITS HDU objects with
:meth:`~spectral_cube.lower_dimensional_objects.Projection.from_hdu`.  Only
FITS reading is currently supported::

    >>> from astropy.io import fits  # doctest: +SKIP
    >>> hdul = fits.open('file.fits')
    >>> projection = Projection.from_hdu(hdul)

Note that if you pass in a :class:`astropy.io.fits.HDUList` object, by default the data will be loaded
from the first HDU; this can be overridden with the ``ext`` keyword.

The resulting :class:`~spectral_cube.lower_dimensional_objects.Projection`
object will have ``.unit``, ``.wcs``, and (if available) ``.beam`` attributes.

If you are working with two dimensional data that have "dummy" third dimensions,
you may load them using the normal :meth:`~spectral_cube.SpectralCube.read` method.
This case is common as such files are normally output from CASA using
``exportfits`` with no additional keywords.  To get a 2D slice, you simply index the
result::

    >>> flat_cube = SpectralCube.read('casa_exported_file.fits')  # doctest: +SKIP
    >>> image = flat_cube[0]


Reading spectra from file
-------------------------

Similar to 2D objects (images), you may want to load 1D slices - spectra - from disk.


You can load :class:`~spectral_cube.lower_dimensional_objects.OneDSpectrum`
objects from FITS HDU objects with
:meth:`~spectral_cube.lower_dimensional_objects.OneDSpectrum.from_hdu`.  As with
projections, only
FITS reading is supported::

    >>> from astropy.io import fits  # doctest: +SKIP
    >>> hdul = fits.open('file.fits')
    >>> projection = OneDSpectrum.from_hdu(hdul)

The spectrum loader only works for 1D spectra with valid FITS WCS in their
headers.  For other types of spectra, you may want to use `specutils
<https://specutils.readthedocs.io/en/stable/>`_ instead.


Direct Initialization
---------------------

If you are interested in directly creating a
:class:`~spectral_cube.SpectralCube` instance, you can do so using a 3-d
Numpy-like array with a 3-d :class:`~astropy.wcs.WCS` object::

    >>> cube = SpectralCube(data=data, wcs=wcs)  # doctest: +SKIP

Here ``data`` can be any Numpy-like array, including *memory-mapped* Numpy
arrays (as mentioned in `Reading from a file`_, memory-mapping is a technique
that avoids reading the whole file into memory and instead accessing it from
the disk as needed).

Hacks for simulated data
------------------------

If you're working on synthetic images or simulated data, where a location on
the sky is not relevant (but the frequency/wavelength axis still is!), a hack
is required to set up the `world coordinate system
<http://docs.astropy.org/en/stable/wcs/>`_.  The header should be set up
such that the projection is cartesian, i.e.::

    CRVAL1 = 0
    CTYPE1 = 'RA---CAR'
    CRVAL2 = 0
    CTYPE2 = 'DEC--CAR'
    CDELT1 = 1.0e-4 //degrees
    CDELT2 = 1.0e-4 //degrees
    CUNIT1 = 'deg'
    CUNIT2 = 'deg'

Note that the x/y axes must always have angular units (i.e., degrees).  If your
data are really in physical units, you should note that in the header in other
comments, but ``spectral-cube`` doesn't care about this.


If the frequency axis is irrelevant, ``spectral-cube`` is probably not the
right tool to use; instead you should use `astropy.io.fits
<http://docs.astropy.org/en/stable/io/fits/>`_ or some other file reader
directly.

