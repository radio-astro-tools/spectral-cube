Spectral Cube documentation
===========================

The ``spectral_cube`` package provides an easy way to read, manipulate,
analyze, and write data cubes with two positional dimensions and one
spectral dimension, optionally with Stokes parameters. The spectral cube
reader is designed to be robust to the wide range of conventions of axis
order, spatial projections, and spectral units that exist, and provides a
uniform interface to all spectral cubes. The package is designed to work with
large files, including files larger than the available memory on a computer.

Overview
--------

The package centers around the
:class:`~spectral_cube.spectral_cube.SpectralCube` class. In the following
sections, we look at how to read data into this class, manipulate spectral
cubes, extract moment maps or subsets of spectral cubes, and write spectral
cubes to files.

Initializing
------------

The :class:`~spectral_cube.spectral_cube.SpectralCube` class is used to
represent 3-dimensional datasets (two positional dimensions and one spectral
dimension) with a World Coordinate System (WCS) projection that describes the
mapping from pixel to world coordinates and vice-versa. The class is imported
with::

    >>> from spectral_cube import SpectralCube

The easiest way to instantiate a
:class:`~spectral_cube.spectral_cube.SpectralCube` is to pass a 3-d numpy
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
:meth:`~spectral_cube.spectral_cube.SpectralCube.read` class method as
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

Masking
-------

In addition to supporting the representation of data and associated WCS, it
is also possible to attach a boolean mask to the
:class:`~spectral_cube.spectral_cube.SpectralCube` class. Masks can take
various forms, but one of the more common ones is a cube with the same
dimensions as the data, and that contains e.g. the boolean value `True` where
data should be used, and the value `False` when the data should be ignored
(though it is also possible to flip the convention around). To create a
boolean mask from a boolean array ``mask_array``, simply use::

    >>> from spectral_cube import SpectralCubeMask
    >>> mask = SpectralCubeMask(mask=mask_array, wcs=cube.wcs)

There are ways to create more efficient masks, and ways to easily combine
masks. This is described in more detail in `Advanced masking`_ below.

Accessing data
--------------

Once you have initialized a :meth:`~spectral_cube.spectral_cube.SpectralCube`
instance, either directly or by reading in a file, there are a number of
ways of accessing the data.

Unmasked data
^^^^^^^^^^^^^

First, you can access the underlying data using the ``data_unmasked`` array
which is a Numpy-like array that has not had the mask applied::

    >>> slice_unmasked = cube.data_unmasked[0,:,:]

.. TODO: show example output

The order of the dimensions of the ``data_unmasked`` array is deterministic -
it is always ``(n_spectral, n_y, n_x)`` irrespective of how the cube was
stored on disk. In the case where the array is memory-mapped, slicing the
array as shown above results in only that slice being read from disk, so it
should be faster than reading the whole dataset.

Masked data
^^^^^^^^^^^

You can also access the masked data using ``data_filled``. This array is a
copy of the original data with any masked value replaced by a fill value
(which is ``np.nan`` by default but can be changed using the ``fill_value``
option in the :class:`~spectral_cube.spectral_cube.SpectralCube`
initializer). The 'filled' data is accessed with e.g.::

    >>> slice_filled = cube.data_filled[0,:,:]

.. TODO: show example output

Note that accessing the filled data should still be efficient because the data
are loaded and filled only once you access the actual data values, so this
should still be efficient for large datasets.

In both the case of the unmasked and filled data, the efficiency breaks down
if you try and access all the data values, for example by doing
``cube.data_filled.sum()``. In such cases, it is more efficient to iterate
over smaller parts of the data (such as slices) rather than access all the
data in one go.

Flattened data
^^^^^^^^^^^^^^

If you are only interested in getting a flat (i.e. 1-d) array of all the
non-masked values, you can also make use of the
:meth:`~spectral_cube.spectral_cube.SpectralCube.flattened` method::

   >>> flat_array = cube.flattened()

.. TODO: show example output

Selecting subsets
-----------------

Extracting a spectral slab
^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a spectral cube, it is easy to extract a sub-cube covering only a subset
of the original range in the spectral axis. To do this, you can use the
:meth:`~spectral_cube.spectral_cube.SpectralCube.spectral_slab` method. This
method takes lower and upper bounds for the spectral axis, as well as an
optional rest frequency, and returns a new
:class:`~spectral_cube.spectral_cube.SpectralCube` instance. The bounds can
be specified as a frequency, wavelength, or a velocity relative to a rest
frequency. If the latter, then the rest frequency needs to be specified. The
bounds and the rest frequency (if applicable) should be given as Astropy
:class:`~astropy.units.Quantity` instances as follows:

    >>> from astropy import units as u
    >>> co_1_0 = cube.spectral_slab(-50 * u.km / u.s, +50 * u.km / u.s,
                                    rest_frequency=115.27120 * u.GHz)

In the above example, regardless of what units the original cube was in, the
:meth:`~spectral_cube.spectral_cube.SpectralCube.spectral_slab` can determine
how to convert the velocities to frequencies if needed. The resulting cube
``co_1_0`` (which is also a
:class:`~spectral_cube.spectral_cube.SpectralCube` instance) then contains
all channels that overlap with the range -50 to 50 km/s relative to the 12CO
1-0 line.

Extracting a sub-cube by indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also easy to extract a sub-cube from pixel coordinates using standard
Numpy slicing notation::

    >>> sub_cube = cube[:100, 10:50, 10:50]

This returns a new :class:`~spectral_cube.spectral_cube.SpectralCube` object
with updated WCS information.

Moment maps and statistics
--------------------------

Producing moment maps from a
:class:`~spectral_cube.spectral_cube.SpectralCube` instance is
straightforward::

    >>> moment_0 = cube.moment(order=0)
    >>> moment_1 = cube.moment(order=1)
    >>> moment_2 = cube.moment(order=2)

By default, moments are computed along the spectral dimension, but it is also
possible to pass the ``axis`` argument to compute them along a different
axis::

    >>> moment_0_along_x = cube.moment(order=0, axis=2)

Advanced masking
----------------

Using a pure boolean array may not always be the most efficient solution,
because it may require a large amount of memory. Other types of mask that can
be used include masks based on simple conditions (e.g. the data values should
be larger than 5) or masks based on the values that they are called with.

Masks based on simple functions that operate on the initial data use the
:class:`~spectral_cube.spectral_cube.LazyMask` class. The motivation behind
the :class:`~spectral_cube.spectral_cube.LazyMask` class is that it is
essentially equivalent to a boolean array, but the boolean values are
computed on-the-fly as needed, meaning that the whole boolean array does not
ever necessarily need to be computed or stored in memory, making it ideal for
very large datasets. The function passed to
:class:`~spectral_cube.spectral_cube.LazyMask` should be a simple function
taking one argument - the dataset itself::

    >>> from spectral_cube import LazyMask
    >>> LazyMask(np.isfinite)

or for example::

    >>> def threshold(data):
    ...     return data > 3.
    >>> LazyMask(threshold)

.. TODO: add example for FunctionalMask

Stokes parameters
-----------------

.. TODO: first we need to make sure the StokesSpectralCube class is working.

Handling large datasets
-----------------------

.. TODO: we can move things specific to large data and copying/referencing here.

Writing out
-----------

You can write out a :class:`~spectral_cube.spectral_cube.SpectralCube`
instance by making use of the
:meth:`~spectral_cube.spectral_cube.SpectralCube.write` method::

    >>> cube.write('new_cube.fits', format='fits')
