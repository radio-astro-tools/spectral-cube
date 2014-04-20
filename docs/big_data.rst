Handling large datasets
=======================

.. TODO: we can move things specific to large data and copying/referencing here.

The :class:`~spectral_cube.SpectralCube` class is designed to allow working
with files larger than can be stored in memory. Numpy supports a
*memory-mapping* mode which means that the data is stored on disk and the
data or subsets of the data are only accessed when needed. The
:mod:`astropy.io.fits` module makes use of this mode by default when reading
in FITS files (though this will not work with compressed - e.g. ``.fits.gz``
- files). The :meth:`~spectral_cube.SpectralCube.read` method will make use
of this feature if available.

If using memory mapping, then When accessing the data using e.g. the
:attr:`~spectral_cube.SpectralCube.data_filled` and
:attr:`~spectral_cube.SpectralCube.data_unmasked` attributes, the data is
still not copied until specific values are accessed. Even taking slices from
these attributes will not read the data until the actual data values are
required.

This efficiency breaks down if you try and access all the data values, for
example by doing ``cube.data_filled.sum()``. In such cases, it is more
efficient to iterate over smaller parts of the data (such as slices) rather
than access all the data in one go, otherwise the whole dataset is loaded
into memory.
