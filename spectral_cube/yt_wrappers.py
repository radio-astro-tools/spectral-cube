import numpy as np


def spectral_cube_to_yt_object(cube, spectral_factor=1.0, center=None, nprocs=1):
    """
    Convert a spectral cube to a yt object that can be further analyzed in yt.

    By default, the yt object returned will be defined in the default yt
    spatial units (1 spatial pixel = 1 cm) centered on the center of the
    spectral cube in all directions. If the ``center`` argument is passed,
    then the cube is still returned in the default yt spatial units, but
    shifted so that the specified coordinates are at the origin in the
    returned object.

    Parameters
    ----------
    cube : `~spectral_cube.spectal_cube.SpectralCube`
        The spectral cube to render
    spectral_factor : float, optional
        Factor by which to stretch the spectral axis. If set to 1, one pixel
        in spectral coordinates is equivalent to one pixel in spatial
        coordinates.
    center : iterable
        Tuple or list containing the three coordinates for the center. These
        should be given as ``(lon, lat, spectral)``.
    """

    from yt.mods import load_uniform_grid

    data = {'density': cube.get_filled_data(fill=0.)}

    nz, ny, nx = cube.shape

    dx = nx / 2.
    dy = ny / 2.
    dz = nz / 2. * spectral_factor

    # Determine center in pixel coordinates
    center = cube.wcs.wcs_world2pix([center], 0)

    pf = load_uniform_grid(data, cube.shape, 1.,
                           bbox=np.array([[-0.5 * spectral_factor, (nz - 0.5) * spectral_factor],
                                          [-0.5, ny - 0.5],
                                          [-0.5, nx - 0.5]]),
                           nprocs=nprocs, periodicity=(False, False, False))

    return pf
