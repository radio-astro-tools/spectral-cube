Masking
=======

Simple boolean masking
----------------------

In addition to supporting the representation of data and associated WCS, it
is also possible to attach a boolean mask to the
:class:`~spectral_cube.SpectralCube` class. Masks can take
various forms, but one of the more common ones is a cube with the same
dimensions as the data, and that contains e.g. the boolean value ``True`` where
data should be used, and the value ``False`` when the data should be ignored
(though it is also possible to flip the convention around). To create a
boolean mask from a boolean array ``mask_array``, simply use::

    >>> from spectral_cube import SpectralCubeMask
    >>> mask = SpectralCubeMask(mask=mask_array, wcs=cube.wcs)

Advanced masking
----------------

Using a pure boolean array may not always be the most efficient solution,
because it may require a large amount of memory. Other types of mask that can
be used include masks based on simple conditions (e.g. the data values should
be larger than 5) or masks based on the values that they are called with.

Masks based on simple functions that operate on the initial data can be
defined using the :class:`~spectral_cube.LazyMask` class. The motivation
behind the :class:`~spectral_cube.LazyMask` class is that it is essentially
equivalent to a boolean array, but the boolean values are computed on-the-fly
as needed, meaning that the whole boolean array does not ever necessarily
need to be computed or stored in memory, making it ideal for very large
datasets. The function passed to :class:`~spectral_cube.LazyMask` should be a
simple function taking one argument - the dataset itself::

    >>> from spectral_cube import LazyMask
    >>> cube = read(...)
    >>> LazyMask(np.isfinite, cube=cube)

or for example::

    >>> def threshold(data):
    ...     return data > 3.
    >>> LazyMask(threshold, cube=cube)

:class:`~spectral_cube.LazyMask` instances can also be defined directly by
specifying conditions on the :class:`~spectral_cube.SpectralCube` instances:

   >>> cube > 5
       LazyMask(...)

Combining and applying masks
----------------------------

Masks can be combined using standard boolean comparison operators::

   >>> background_mask = valid_mask & ~signal_mask

The ``&`` operator is used as an *and* operator, the ``|`` operator is used
as an *or* operator, and the ``~`` operator is used to indicate the *not*
logical operator.

To apply a new mask to a :class:`~spectral_cube.SpectralCube` class, use the
:meth:`~spectral_cube.SpectralCube.with_mask` method, which can take a mask
and combine it with any pre-existing mask::

    >>> cube2 = cube.with_mask(new_mask)

In the above example, ``cube2`` contains a mask that is the ``&`` combination
of ``new_mask`` with the existing mask on ``cube``. The ``cube2`` object
contains a view to the same data as ``cube``, so no data is copied during
this operation.

Fill values
-----------

When accessing the data (see :doc:`accessing`), the mask may be applied to
the data and the masked values replaced by a *fill* value. This fill value
can be set using the ``fill_value`` initializer in
:class:`~spectral_cube.SpectralCube`, and is set to ``np.nan`` by default. To
change the fill value on a cube, you can make use of the
:meth:`~spectral_cube.SpectralCube.with_fill_value` method::

    >>> cube2 = cube.with_fill_value(0.)

This returns a new :class:`~spectral_cube.SpectralCube` instance that
contains a view to the same data in ``cube`` (so no data are copied).

Inclusion and Exclusion
-----------------------

The term "mask" is often used to refer both to the act of exluding
and including pixels from analysis. To be explicit about how they behave,
all mask objects have an
:meth:`~spectral_cube.masks.MaskBase.include` method that returns a boolean
array. True values in this array indicate that the pixel is included/valid,
and not filtered/replaced in any way. Conversely, True values in the output
from :meth:`~spectral_cube.masks.MaskBase.exclude`
indicate the pixel is excluded/invalid, and will be filled/filtered.
The inclusion/exclusion behavior of any mask can be inverted via
``mask_inverse = ~mask``.

.. TODO: add example for FunctionalMask
