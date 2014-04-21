Accessing data
==============

Once you have initialized a :meth:`~spectral_cube.SpectralCube`
instance, either directly or by reading in a file, there are a number of
ways of accessing the data.

Unmasked data
-------------

First, you can access the underlying data using the ``data_unmasked`` array
which is a Numpy-like array that has not had the mask applied::

    >>> slice_unmasked = cube.data_unmasked[0,:,:]

.. TODO: show example output

The order of the dimensions of the ``data_unmasked`` array is deterministic -
it is always ``(n_spectral, n_y, n_x)`` irrespective of how the cube was
stored on disk.

Masked data
-----------

You can also access the masked data using ``data_filled``. This array is a
copy of the original data with any masked value replaced by a fill value
(which is ``np.nan`` by default but can be changed using the ``fill_value``
option in the :class:`~spectral_cube.SpectralCube`
initializer). The 'filled' data is accessed with e.g.::

    >>> slice_filled = cube.data_filled[0,:,:]

.. TODO: show example output

Note that accessing the filled data should still be efficient because the data
are loaded and filled only once you access the actual data values, so this
should still be efficient for large datasets.

Flattened data
--------------

If you are only interested in getting a flat (i.e. 1-d) array of all the
non-masked values, you can also make use of the
:meth:`~spectral_cube.SpectralCube.flattened` method::

   >>> flat_array = cube.flattened()

.. TODO: show example output