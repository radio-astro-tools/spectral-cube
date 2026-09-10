.. _doc_handling_large_datasets:

Handling large datasets
=======================
.. currentmodule:: spectral_cube

.. TODO: we can move things specific to large data and copying/referencing here.

The :class:`SpectralCube` class is designed to allow working
with files larger than can be stored in memory. To take advantage of this and
work effectively with large spectral cubes, you should keep the following
three ideas in mind:

- Work with small subsets of data at a time.
- Minimize data copying.
- Minimize the number of passes over the data.
- Specifying temporary file directory.

Work with small subsets of data at a time
-----------------------------------------

Numpy supports a *memory-mapping* mode which means that the data is stored
on disk and the array elements are only loaded into memory when needed.
``spectral_cube`` takes advantage of this if possible, to avoid loading
large files into memory.

Typically, working with NumPy involves writing code that operates
on an entire array at once. For example::

    x = <a numpy array>
    y = np.sum(np.abs(x * 3 + 10), axis=0)

Unfortunately, this code creates several temporary arrays
whose size is equal to ``x``. This is infeasible if ``x`` is a large
memory-mapped array, because an operation like ``(x * 3)`` will require
more RAM than exists on your system. A better way to compute y is
by working with a single slice of ``x`` at a time::

    y = np.zeros_like(x[0])
    for plane in x:
        y += np.abs(plane * 3 + 10)

Many methods in :class:`SpectralCube` allow you to extract subsets
of relevant data, to make writing code like this easier:

- `~spectral_cube.base_class.MaskableArrayMixinClass.filled_data`,
  `~spectral_cube.BaseSpectralCube.unmasked_data`,
  `~spectral_cube.base_class.SpatialCoordMixinClass.world` all accept Numpy style
  slice syntax. For
  example, ``cube.filled_data[0:3, :, :]`` returns only the first 3 spectral
  channels of the cube, with masked elements replaced with ``cube.fill_value``.
- :class:`~spectral_cube.SpectralCube` itself can be sliced to extract subcubes
- `~spectral_cube.base_class.BaseSpectralCube.spectral_slab` extracts a subset
  of spectral channels.

Many methods in :class:`~spectral_cube.SpectralCube` iterate over smaller
chunks of data, to avoid large memory allocations when working with big cubes.
Some of these have a ``how`` keyword parameter, for fine-grained control over
how much memory is accessed at once.  ``how='cube'`` works with the entire
array in memory, ``how='slice'`` works with one slice at a time, and
``how='ray'`` works with one ray at a time.

As a user, your best strategy for working with large datasets is to rely on
builtin methods to :class:`~spectral_cube.SpectralCube`, and to access data
from `~spectral_cube.base_class.MaskableArrayMixinClass.filled_data` and
`~spectral_cube.BaseSpectralCube.unmasked_data` in smaller chunks if
possible.


.. warning ::
    At the moment, :meth:`~SpectralCube.argmax` and :meth:`~SpectralCube.argmin`,
    are **not** optimized for handling large datasets.


Minimize Data Copying
---------------------

Methods in :class:`~spectral_cube.SpectralCube` avoid copying as much as
possible. For example, all of the following operations create new cubes or
masks without copying any data::

    >>> mask = cube > 3  # doctest: +SKIP
    >>> slab = cube.spectral_slab(...)  # doctest: +SKIP
    >>> subcube = cube[0::2, 10:, 0:30]  # doctest: +SKIP
    >>> cube2 = cube.with_fill(np.nan)  # doctest: +SKIP
    >>> cube2 = cube.apply_mask(mask)  # doctest: +SKIP

Minimize the number of passes over the data
-------------------------------------------

Accessing memory-mapped arrays is much slower than a normal
array, due to the overhead of reading from disk. Because of this,
it is more efficient to perform computations that iterate over the
data as few times as possible.

An even subtler issue pertains to how the 3D or 4D spectral cube
is arranged as a 1D sequence of bytes in a file. Data access is much faster
when it corresponds to a single contiguous scan of bytes on disk.
For more information on this topic, see `this tutorial on Numpy strides
<http://scipy-lectures.github.io/advanced/advanced_numpy/#indexing-scheme-strides>`_.

Recipe for large cube operations that can't be done in memory
-------------------------------------------------------------

Sometimes, you will need to run full-cube operations that can't be done in
memory and can't be handled by spectral-cube's built in operations.
An example might be converting your cube from Jy/beam to K when you have
a very large (e.g., >10GB) cube.

Handling this sort of situation requires several manual steps. First, hard
drive space needs to be allocated for the output data.  Then, the cube
must be manually looped over using a strategy that holds only limited data
in memory.::

    >>> import shutil # doctest: +SKIP
    >>> from spectral_cube import SpectralCube # doctest: +SKIP
    >>> from astropy.io import fits # doctest: +SKIP

    >>> cube = SpectralCube.read('file.fits') # doctest: +SKIP

    >>> # this copy step is necessary to allocate memory for the output
    >>> shutil.copy('file.fits', 'newfile.fits') # doctest: +SKIP
    >>> outfh = fits.open('newfile.fits', mode='update') # doctest: +SKIP

    >>> jtok_factors = cube.jtok_factors() # doctest: +SKIP
    >>> for index,(slice,factor) in enumerate(zip(cube,factors)): # doctest: +SKIP
    ...     outfh[0].data[index] = slice * factor # doctest: +SKIP
    ...     outfh.flush() # write the data to disk # doctest: +SKIP
    >>> outfh[0].header['BUNIT'] = 'K' # doctest: +SKIP
    >>> outfh.flush() # doctest: +SKIP


Specifying temporary file directory
-----------------------------------

By default, spectral-cube uses memory-mapped arrays to store intermediate results, which are saved
in the system's temporary directory. Some systems may have limited space for temporary files, however.

To specify a different directory, you can set the environment variable ``TMPDIR`` to the desired path::

    >>> TMPDIR='/path/to/directory' python spectral_cube_script.py # doctest: +SKIP

Within a python environment, you can instead set the ``memmap_dir`` keyword for calls to individual methods::

    >>> cube = SpectralCube.read('file.fits') # doctest: +SKIP
    >>> convolved_cube = cube.convolve_to(common_beam, memmap_dir='/path/to/directory') # doctest: +SKIP

