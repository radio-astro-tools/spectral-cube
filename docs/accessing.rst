Accessing data
==============

Once you have initialized a :meth:`~spectral_cube.SpectralCube`
instance, either directly or by reading in a file, you can easily access the
data values and the world coordinate information.

Data values
-----------

You can access the underlying data using the ``unmasked_data`` array which is
a Numpy-like array::

    >>> slice_unmasked = cube.unmasked_data[0,:,:]

The order of the dimensions of the ``unmasked_data`` array is deterministic -
it is always ``(n_spectral, n_y, n_x)`` irrespective of how the cube was
stored on disk.

.. note:: The term ``unmasked`` indicates that the data is the raw original
          data from the file. :meth:`~spectral_cube.SpectralCube` also allows
          masking of values, which is discussed in :doc:`masking`.

World coordinates
-----------------

Given a cube object, it is straightforward to find the coordinates along the
spectral axis::

   >>> cube.spectral_axis
   [ -2.97198762e+03  -2.63992044e+03  -2.30785327e+03  -1.97578610e+03
     -1.64371893e+03  -1.31165176e+03  -9.79584583e+02  -6.47517411e+02
     ...
      3.15629983e+04   3.18950655e+04   3.22271326e+04   3.25591998e+04
      3.28912670e+04   3.32233342e+04] m / s

The default units of a spectral axis are determined from the FITS header or
WCS object used to initialize the cube, but it is also possible to change the
spectral axis (see :doc:`manipulating`).

More generally, it is possible to extract the world coordinates of all the
pixels using the :attr:`~spectral_cube.SpectralCube.world` property, which
returns the spectral axis then the two positional coordinates in reverse
order (in the same order as the data indices).

   >>> velo, dec, ra = cube.world[:]

In order to extract coordinates, a slice (such as ``[:]`` above) is required.
Using ``[:]`` will return three 3-d arrays with the coordinates for all
pixels. Using e.g. ``[0,:,:]`` will return three 2-d arrays of coordinates for
the first spectral slice.

If you forget to specify a slice, you will get the following error:

   >>> velo, dec, ra = cube.world
   ...
   Exception: You need to specify a slice (e.g. ``[:]`` or ``[0,:,:]`` in order to access this property.

In the case of large data cubes, requesting the coordinates of all pixels
would likely be too slow, so the slicing allows you to compute only a subset
of the pixel coordinates (see :doc:`big_data` for more information on dealing
with large data cubes).

