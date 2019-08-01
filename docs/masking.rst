Masking
=======

Getting started
---------------

In addition to supporting the representation of data and associated WCS, it
is also possible to attach a boolean mask to the
:class:`~spectral_cube.SpectralCube` class. Masks can take
various forms, but one of the more common ones is a cube with the same
dimensions as the data, and that contains e.g. the boolean value `True` where
data should be used, and the value `False` when the data should be ignored
(though it is also possible to flip the convention around; see
:ref:`mask_inclusion_exclusion`). To create a
boolean mask from a boolean array ``mask_array``, you can for example use::

    >>> from astropy import units as u
    >>> from spectral_cube import BooleanArrayMask
    >>> mask = BooleanArrayMask(mask=mask_array, wcs=cube.wcs)  # doctest: +SKIP

.. note::

   Currently, the mask convention is opposite of what is defined for
   Numpy masked array and Astropy ``Table``.

Using a pure boolean array may not always be the most efficient solution,
because it may require a large amount of memory.

You can also create a mask using simple conditions directly on the cube
values themselves, for example::

    >>> mask = cube > 1.3*u.K  # doctest: +SKIP

This is more efficient, because the condition is actually evaluated on-the-fly
as needed.  Note that units equivalent to the cube's units must be used.

Masks can be combined using standard boolean comparison operators::

   >>> new_mask = (cube > 1.3*u.K) & (cube < 100.*u.K)  # doctest: +SKIP

The available operators are ``&`` (and), ``|`` (or), and ``~`` (not).

To apply a new mask to a :class:`~spectral_cube.SpectralCube` class, use the
:meth:`~spectral_cube.SpectralCube.with_mask` method, which can take a mask
and combine it with any pre-existing mask::

    >>> cube2 = cube.with_mask(new_mask)  # doctest: +SKIP

In the above example, ``cube2`` contains a mask that is the ``&`` combination
of ``new_mask`` with the existing mask on ``cube``. The ``cube2`` object
contains a view to the same data as ``cube``, so no data is copied during
this operation.

Boolean arrays can also be used as input to
:meth:`~spectral_cube.SpectralCube.with_mask`, assuming the shape of the mask
and the data match::

    >>> cube2 = cube.with_mask(boolean_array)  # doctest: +SKIP

Any boolean area that can be `broadcast
<http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ to the cube
shape can be used as a boolean array mask.

Accessing masked data
---------------------

As mention in :doc:`accessing`, the raw and unmasked data can be accessed with
the `spectral_cube.spectral_cube.BaseSpectralCube.unmasked_data`
attribute.  You can access the masked data using ``filled_data``. This array is
a copy of the original data with any masked value replaced by a fill value
(which is ``np.nan`` by default but can be changed using the ``fill_value``
option in the :class:`~spectral_cube.SpectralCube` initializer). The 'filled'
data is accessed with e.g.::

    >>> slice_filled = cube.filled_data[0,:,:]  # doctest: +SKIP

Note that accessing the filled data should still be efficient because the data
are loaded and filled only once you access the actual data values, so this
should still be efficient for large datasets.

If you are only interested in getting a flat (i.e. 1-d) array of all the
non-masked values, you can also make use of the
:meth:`~spectral_cube.SpectralCube.flattened` method::

   >>> flat_array = cube.flattened()  # doctest: +SKIP

Fill values
-----------

When accessing the data (see :doc:`accessing`), the mask may be applied to
the data and the masked values replaced by a *fill* value. This fill value
can be set using the ``fill_value`` initializer in
:class:`~spectral_cube.SpectralCube`, and is set to ``np.nan`` by default. To
change the fill value on a cube, you can make use of the
:meth:`~spectral_cube.SpectralCube.with_fill_value` method::

    >>> cube2 = cube.with_fill_value(0.)  # doctest: +SKIP

This returns a new :class:`~spectral_cube.SpectralCube` instance that
contains a view to the same data in ``cube`` (so no data are copied).

.. _mask_inclusion_exclusion:

Inclusion and Exclusion
-----------------------

The term "mask" is often used to refer both to the act of exluding
and including pixels from analysis. To be explicit about how they behave,
all mask objects have an
:meth:`~spectral_cube.masks.MaskBase.include` method that returns a boolean
array. `True` values in this array indicate that the pixel is included/valid,
and not filtered/replaced in any way. Conversely, `True` values in the output
from :meth:`~spectral_cube.masks.MaskBase.exclude`
indicate the pixel is excluded/invalid, and will be filled/filtered.
The inclusion/exclusion behavior of any mask can be inverted via::

    >>> mask_inverse = ~mask  # doctest: +SKIP

Advanced masking
----------------

Masks based on simple functions that operate on the initial data can be
defined using the :class:`~spectral_cube.LazyMask` class. The motivation
behind the :class:`~spectral_cube.LazyMask` class is that it is essentially
equivalent to a boolean array, but the boolean values are computed on-the-fly
as needed, meaning that the whole boolean array does not ever necessarily
need to be computed or stored in memory, making it ideal for very large
datasets. The function passed to :class:`~spectral_cube.LazyMask` should be a
simple function taking one argument - the dataset itself::

    >>> from spectral_cube import LazyMask
    >>> cube = read(...)  # doctest: +SKIP
    >>> LazyMask(np.isfinite, cube=cube)  # doctest: +SKIP

or for example::

    >>> def threshold(data):
    ...     return data > 3.
    >>> LazyMask(threshold, cube=cube)  # doctest: +SKIP

As shown in `Getting Started`_, :class:`~spectral_cube.LazyMask` instances
can also be defined directly by specifying conditions on
:class:`~spectral_cube.SpectralCube` objects:

   >>> cube > 5*u.K  # doctest: +SKIP
   LazyComparisonMask(...)

.. TODO: add example for FunctionalMask


Outputting masks
----------------

The attached mask to the given :class:`~spectral_cube.SpectralCube` class can
be converted into a CASA image using :func:`~spectral_cube.io.casa_masks.make_casa_mask`:

  >>> from spectral_cube.io.casa_masks import make_casa_mask
  >>> make_casa_mask(cube, 'casa_mask.image', add_stokes=False)  # doctest: +SKIP

Optionally, a redundant Stokes axis can be added to match the original CASA
image.

.. Masks may also be appended to an existing CASA image::
..   >>> make_casa_mask(cube, 'casa_mask.image', append_to_img=True,
..                      img='casa.image')

.. note::
    Outputting to CASA masks requires that `spectral_cube` be run from a CASA python session.

Masking cubes with other cubes
------------------------------

A common use case is to mask a cube based on another cube in the same
coordinates.  For example, you want to create a mask of 13CO based on the
brightness of 12CO.  This can be done straightforwardly if they are on an
identical grid::

    >>> mask_12co = cube12co > 0.5*u.Jy  # doctest: +SKIP
    >>> masked_cube13co = cube13co.with_mask(mask_12co)  # doctest: +SKIP

If you see errors such as ``WCS does not match mask WCS``, but you're confident
that your two cube are on the same grid, you should have a look at the
``cube.wcs`` attribute and see if there are subtle differences in the world
coordinate parameters.  These frequently occur when converting from frequency
to velocity as there is inadequate precision in the rest frequency.

For example, these two axes are *nearly* identical, but not perfectly so::

    Number of WCS axes: 3
    CTYPE : 'RA---SIN'  'DEC--SIN'  'VRAD'
    CRVAL : 269.08866286689999  -21.956244813729999  -3000.000559989533
    CRPIX : 161.0  161.0  1.0
    PC1_1 PC1_2 PC1_3  : 1.0  0.0  0.0
    PC2_1 PC2_2 PC2_3  : 0.0  1.0  0.0
    PC3_1 PC3_2 PC3_3  : 0.0  0.0  1.0
    CDELT : -1.3888888888889999e-05  1.3888888888889999e-05  299.99999994273281
    NAXIS    : 0 0

    Number of WCS axes: 3
    CTYPE : 'RA---SIN'  'DEC--SIN'  'VRAD'
    CRVAL : 269.08866286689999  -21.956244813729999  -3000.0000242346514
    CRPIX : 161.0  161.0  1.0
    PC1_1 PC1_2 PC1_3  : 1.0  0.0  0.0
    PC2_1 PC2_2 PC2_3  : 0.0  1.0  0.0
    PC3_1 PC3_2 PC3_3  : 0.0  0.0  1.0
    CDELT : -1.3888888888889999e-05  1.3888888888889999e-05  300.00000001056611
    NAXIS    : 0 0

In order to compose masks from these, we need to set the ``wcs_tolerance`` parameter::

    >>> masked_cube13co = cube13co.with_mask(mask_12co, wcs_tolerance=1e-3)  # doctest: +SKIP

which in this case will check equality at the 1e-3 level, which truncates
the 3rd CRVAL to the point of equality before comparing the values.
