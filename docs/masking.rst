Masking
=======

Simple boolean masking
----------------------

In addition to supporting the representation of data and associated WCS, it
is also possible to attach a boolean mask to the
:class:`~spectral_cube.SpectralCube` class. Masks can take
various forms, but one of the more common ones is a cube with the same
dimensions as the data, and that contains e.g. the boolean value `True` where
data should be used, and the value `False` when the data should be ignored
(though it is also possible to flip the convention around). To create a
boolean mask from a boolean array ``mask_array``, simply use::

    >>> from spectral_cube import SpectralCubeMask
    >>> mask = SpectralCubeMask(mask=mask_array, wcs=cube.wcs)

There are ways to create more efficient masks, and ways to easily combine
masks. This is described in more detail in `Advanced masking`_ below.

Advanced masking
----------------

Using a pure boolean array may not always be the most efficient solution,
because it may require a large amount of memory. Other types of mask that can
be used include masks based on simple conditions (e.g. the data values should
be larger than 5) or masks based on the values that they are called with.

Masks based on simple functions that operate on the initial data use the
:class:`~spectral_cube.LazyMask` class. The motivation behind
the :class:`~spectral_cube.LazyMask` class is that it is
essentially equivalent to a boolean array, but the boolean values are
computed on-the-fly as needed, meaning that the whole boolean array does not
ever necessarily need to be computed or stored in memory, making it ideal for
very large datasets. The function passed to
:class:`~spectral_cube.LazyMask` should be a simple function
taking one argument - the dataset itself::

    >>> from spectral_cube import LazyMask
    >>> LazyMask(np.isfinite)

or for example::

    >>> def threshold(data):
    ...     return data > 3.
    >>> LazyMask(threshold)

.. TODO: add example for FunctionalMask