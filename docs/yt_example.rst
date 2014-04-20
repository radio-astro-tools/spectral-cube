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

    >>> pf = cube.to_yt(spectral_factor=0.5)

The ``pf`` object is then a yt object that can be used for rendering! By
default the dataset is defined in pixel coordinates, going from -n/2 to n/2 (to ensure that the cube is centered on the origin by default).

It is possible to specify a different center for the cube by passing the ``center`` argument, which should be a tuple of (x, y, spectral) coordinates. If the coordinates are passed without units, then they are assumed to be pixel coordinates. If the values passed are Astropy `~astropy.units.Quantity` objects with units, then the values are assumed to be world coordinates. For example:

    >>> from astropy import units as u
    >>> pf = cube.to_yt(spectral_factor=0.5, center=(51.424522 * u.deg,
                                                     30.723611 * u.deg,
                                                     5205.18071 * u.m / u.s])

will ensure the cube is centered at the specified coordinates. The cube will still be defined in pixel coordinates in yt, but the origin will be the one corresponding to the above coordinates.

.. TODO: add a way to center it on a specific coordinate and return in world
.. coordinate offset.

Visualization example
---------------------

This section shows an example of a rendering script that can be used to
produce a 3-d isocontour visualization using an object returned by
:meth:`~spectral_cube.SpectralCube.to_yt`::

    from yt.mods import ColorTransferFunction, write_bitmap

    # Extract the yt object from the SpectralCube instance
    pf = cube.to_yt(spectral_factor=0.5)

    # Set the number of levels, the minimum and maximum level and the width
    # of the isocontours
    n_v = 10
    vmin = 0.3
    vmax = 2.0
    dv = 0.01

    # Set up color transfer function
    transfer = ColorTransferFunction((vmin, vmax))
    transfer.add_layers(n_v, dv, colormap='RdBu_r')

    # Set up the camera parameters
    center = [0., 0., 0.]  # pixel units relative to current center
    direction = np.array([0.0, 1.0, 0.0])
    width = 1000.  # pixels

    camera = pf.h.camera(center, direction, width, size, transfer,
                         fields=['flux'])

    # Take a snapshot and save to a file
    snapshot = camera.snapshot()
    write_bitmap(snapshot, 'cube_rendering.png', transpose=True)
