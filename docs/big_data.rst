Handling large datasets
=======================
.. currentmodule:: spectral_cube

.. TODO: we can move things specific to large data and copying/referencing here.

The :class:`SpectralCube` class is designed to allow working
with files larger than can be stored in memory. Numpy supports a
*memory-mapping* mode which means that the data is stored on disk and the
data or subsets of the data are only accessed when needed. The
:mod:`astropy.io.fits` module makes use of this mode by default when reading
in FITS files (though this will not work with compressed - e.g. ``.fits.gz``
- files). The :meth:`~SpectralCube.read` method will make use
of this feature if available.

Functions in spectral-cube are designed to minimize the amount of data
that is read into memory at once. For example, an operation like::

    >>> cube.filled_data[0:5, 0:3, 0:2]

Only extracts the values in a ``6x4x3`` subcube. The :meth:`~SpectralCube.world`
and :meth:`~SpectralCube.unmasked_data` methods also behave this way.

Other functions return views into datasets without copying them, and are also fast.
For example, all of the following operations are fast::

    >>> mask = cube > 3   # creates a lazily-evaluated mask
    >>> slab = cube.spectral_slab(...)
    >>> subcube = cube[0::2, 10:, 0:30]
    >>> cube2 = cube.with_fill(np.nan)
    >>> cube2 = cube.apply_mask(mask)

Some functions, like :meth:`~SpectralCube.moment`,
require scanning over the full dataset, and are thus time consuming. However,
as much as possible these functions iterate over smaller chunks of data, to
prevent running out of memory when working with large cubes.

As a user, your best strategy for working with large datasets is to rely on
builtin-methods to :class:`SpectralCube`, and to access data from
:meth:`~SpectralCube.filled_data` and :meth:`~SpectralCube.unmasked_data`
in smaller chunks if possible.

.. warning ::
    At the moment, the :meth:`~SpectralCube.max`, :meth:`~SpectralCube.min`,
    :meth:`~SpectralCube.argmax`, :meth:`~SpectralCube.argmin`,
    and :meth:`~SpectralCube.sum` methods are **not** optimized for handling
    large datasets.
