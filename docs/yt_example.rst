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

    >>> ds = cube.to_yt(spectral_factor=0.5)

The ``ds`` object is then a yt object that can be used for rendering! By
default the dataset is defined in pixel coordinates, going from ``0.5`` to ``n+0.5``,
as would be the case in ds9, for example. Along the spectral axis, this range
will be modified if ``spectral_factor`` does not equal unity.

When working with datasets in yt, it may be useful to convert world coordinates
to pixel coordinates, so that whenever you may have to input a position in yt
(e.g., for slicing or volume rendering) you can get the pixel coordinate that
corresponds to the desired world coordinate. For this purpose, the method
:meth:`~spectral_cube.SpectralCube.world2yt` is provided::

    >>> import astropy.units as u
    >>> pix_coord = cube.world2yt([51.424522, # deg
                                   30.723611, # deg
                                   5205.18071 # m/s])

which handles a non-unity ``spectral_factor`` automatically if it was included in the
call to :meth:`~spectral_cube.SpectralCube.to_yt`.

There is also a reverse method provided, :meth:`~spectral_cube.SpectralCube.yt2world`::

    >>> world_coord = cube.yt2world([ds.domain_center])

which in this case would return the world coordinates of the center of the dataset
in yt.

.. TODO: add a way to center it on a specific coordinate and return in world
.. coordinate offset.

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
    pf = cube.to_yt(spectral_factor=0.75)

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
    center = cube.world2yt([51.424522 * u.deg,
                            30.723611 * u.deg,
                            5205.18071 * u.m / u.s])
    direction = np.array([1.0, 0.0, 0.0])
    width = 100.  # pixels
    size = 1024

    camera = pf.h.camera(center, direction, width, size, transfer,
                         fields=['flux'])

    # Take a snapshot and save to a file
    snapshot = camera.snapshot()
    write_bitmap(snapshot, 'cube_rendering.png', transpose=True)