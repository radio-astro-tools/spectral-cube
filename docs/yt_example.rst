Visualizing spectral cubes with yt
==================================

Extracting yt objects
---------------------

The :class:`~spectral_cube.SpectralCube` class includes a
:meth:`~spectral_cube.SpectralCube.to_yt` method that makes is easy to return
an object that can be used by `yt <http://yt-project.org>`_ to make volume
renderings or other visualizations of the data. One common issue with volume
rendering of spectral cubes is that you may not want pixels along the
spectral axis to be given the same '3-d' size as positional pixels, so the
:meth:`~spectral_cube.SpectralCube.to_yt` method includes a
``spectral_factor`` argument that can be used to compress or expand the
spectral axis.

The :meth:`~spectral_cube.SpectralCube.to_yt` method is used as follows::

    >>> ytcube = cube.to_yt(spectral_factor=0.5)  # doctest: +SKIP
    >>> ds = ytcube.dataset  # doctest: +SKIP

.. WARNING:: The API change in
   https://github.com/radio-astro-tools/spectral-cube/pull/129 affects the
   interpretation of the 0-pixel.  There may be a 1-pixel offset between the yt
   cube and the SpectralCube

The ``ds`` object is then a yt object that can be used for rendering! By
default the dataset is defined in pixel coordinates, going from ``0.5`` to
``n+0.5``, as would be the case in ds9, for example. Along the spectral axis,
this range will be modified if ``spectral_factor`` does not equal unity.

When working with datasets in yt, it may be useful to convert world coordinates
to pixel coordinates, so that whenever you may have to input a position in yt
(e.g., for slicing or volume rendering) you can get the pixel coordinate that
corresponds to the desired world coordinate. For this purpose, the method
:meth:`~spectral_cube.ytCube.world2yt` is provided::

    >>> import astropy.units as u
    >>> pix_coord = ytcube.world2yt([51.424522,
    ...                              30.723611,
    ...                              5205.18071],  # units of deg, deg, m/s
    ...                             )  # doctest: +SKIP

There is also a reverse method provided, :meth:`~spectral_cube.ytCube.yt2world`::

    >>> world_coord = ytcube.yt2world([ds.domain_center])  # doctest: +SKIP

which in this case would return the world coordinates of the center of the dataset
in yt.

.. TODO: add a way to center it on a specific coordinate and return in world
.. coordinate offset.

.. note::

    The :meth:`~spectral_cube.SpectralCube.to_yt` method and its associated
    coordinate methods are compatible with both yt v. 2.x and v. 3.0 and
    following, but use of version 3.0 or later is recommended due to
    substantial improvements in support for FITS data. For more information on
    how yt handles FITS datasets, see `the yt docs
    <http://yt-project.org/docs/3.0/examining/loading_data.html#fits-data>`_.

Visualization example
---------------------

This section shows an example of a rendering script that can be used to
produce a 3-d isocontour visualization using an object returned by
:meth:`~spectral_cube.SpectralCube.to_yt`::

    import numpy as np
    from spectral_cube import read
    from yt.mods import ColorTransferFunction, write_bitmap
    import astropy.units as u

    # Read in spectral cube
    cube = read('L1448_13CO.fits', format='fits')

    # Extract the yt object from the SpectralCube instance
    ytcube = cube.to_yt(spectral_factor=0.75)
    ds = ytcube.dataset

    # Set the number of levels, the minimum and maximum level and the width
    # of the isocontours
    n_v = 10
    vmin = 0.05
    vmax = 4.0
    dv = 0.02

    # Set up color transfer function
    transfer = ColorTransferFunction((vmin, vmax))
    transfer.add_layers(n_v, dv, colormap='RdBu_r')

    # Set up the camera parameters

    # Derive the pixel coordinate of the desired center
    # from the corresponding world coordinate
    center = ytcube.world2yt([51.424522,
                              30.723611,
                              5205.18071])
    direction = np.array([1.0, 0.0, 0.0])
    width = 100.  # pixels
    size = 1024

    camera = ds.h.camera(center, direction, width, size, transfer,
                         fields=['flux'])

    # Take a snapshot and save to a file
    snapshot = camera.snapshot()
    write_bitmap(snapshot, 'cube_rendering.png', transpose=True)

You can move the camera around; see the `yt camera docs
<http://yt-project.org/docs/dev/reference/api/generated/yt.visualization.volume_rendering.camera.Camera.html>`_.

Movie Making
------------

There is a simple utility for quick movie making.  The default movie is a rotation
of the cube around one of the spatial axes, going from PP -> PV space and back.::

    >>> cube = read('cube.fits', format='fits')  # doctest: +SKIP
    >>> ytcube = cube.to_yt()  # doctest: +SKIP
    >>> images = ytcube.quick_render_movie('outdir')  # doctest: +SKIP

The movie only does rotation, but it is a useful stepping-stone if you wish to
learn how to use yt's rendering system.

Example:

.. raw:: html

   <iframe src="http://player.vimeo.com/video/104489207" width=500 height=281
   frameborder=0 webkitallowfullscreen mozallowfullscreen allowfullscreen>
   </iframe>

SketchFab Isosurface Contours
-----------------------------

For data exploration, making movies can be tedious - it is difficult to control
the camera and expensive to generate new renderings.  Instead, creating a 'model'
from the data and exporting that to SketchFab can be very useful.  Only
grayscale figures will be created with the quicklook code.

You need an account on sketchfab.com for this to work.::

   >>> ytcube.quick_isocontour(title='GRS l=49 13CO 1 K contours', level=1.0)  # doctest: +SKIP


Here's an example:

.. raw:: html

   <iframe width="640" height="480" src="https://sketchfab.com/models/4933bb846b374e71a2765373a0be9fef/embed" frameborder="0" allowfullscreen mozallowfullscreen="true" webkitallowfullscreen="true" onmousewheel=""></iframe>

   <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;">
       <a href="https://sketchfab.com/models/4933bb846b374e71a2765373a0be9fef" style="font-weight: bold; color: #1CAAD9;">GRS l=49 13CO 1 K contours</a>
       by <a href="https://sketchfab.com/keflavich" style="font-weight: bold; color: #1CAAD9;">keflavich</a>
       on <a href="https://sketchfab.com" style="font-weight: bold; color: #1CAAD9;">Sketchfab</a>
   </p>

You can also export locally to .ply and .obj files, which can be read by many
programs (sketchfab, meshlab, blender).  See the `yt page
<http://yt-project.org/doc/visualizing/sketchfab.html>`_ for details.::

   >>> ytcube.quick_isocontour(export_to='ply', filename='meshes.ply', level=1.0)  # doctest: +SKIP
   >>> ytcube.quick_isocontour(export_to='obj', filename='meshes', level=1.0)  # doctest: +SKIP
