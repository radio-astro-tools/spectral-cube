import numpy as np


def spectral_cube_to_yt_object(cube, spectral_factor=1.0, nprocs=1):
    """
    Convert a spectral cube to a yt object that can be further analyzed in yt.

    Parameters
    ----------
    cube : `~spectral_cube.spectal_cube.SpectralCube`
        The spectral cube to render
    spectral_factor : float, optional
        Factor by which to stretch the spectral axis. If set to 1, one pixel
        in spectral coordinates is equivalent to one pixel in spatial
        coordinates.
    """

    from yt.mods import load_uniform_grid

    data = {'density': cube.get_filled_data(fill=0.)}

    nz, ny, nx = cube.shape

    dx = nx / 2.
    dy = ny / 2.
    dz = nz / 2. * spectral_factor

    pf = load_uniform_grid(data, cube.shape, 1.,
                           bbox=np.array([[-0.5 * spectral_factor, (nz - 0.5) * spectral_factor],
                                          [-0.5, ny - 0.5],
                                          [-0.5, nx - 0.5]]),
                           nprocs=nprocs, periodicity=(False, False, False))

    return pf


def render_cube_in_yt(cube, filename, spectral_factor=1.0, nprocs=1,
                      vmin=None, vmax=None, n_levels=10, level_width=None,
                      fly_around=False, center=None, zoom=1., size=512):
    """
    Render a spectral cube using yt.

    Parameters
    ----------
    cube : `~spectral_cube.spectal_cube.SpectralCube`
        The spectral cube to render
    filename : str
        The name of the output file
    spectral_factor : float, optional
        Factor by which to stretch the spectral axis. If set to 1, one pixel
        in spectral coordinates is equivalent to one pixel in spatial
        coordinates.
    nprocs : int, optional
        Number of processors to use for the rendering
    vmin : float, optional
        Minimum level for the iso-contours
    vmax : float, optional
        Maximum level for the iso-contours
    n_levels : int, optional
        Number of iso-contours
    level_width : float, optional
        The width of the isocontours
    fly_around : int, optional
        If set, generate this number of frames to fly around the spectral cube.
    center : tuple or list of float, optional
        The world coordinates to center on - in the native world coordinates
        of the spectral cube.
    zoom : float, optional
        Zoom factor relative to default zoom
    size : int, optional
        The image size
    """

    from yt.mods import ColorTransferFunction, write_bitmap

    pf = spectral_cube_to_yt_object(cube, spectral_factor=spectral_factor, nprocs=nprocs)

    # Auto-determine levels if needed
    if vmin is None:
        vmin = data['density'][data['density'] > 0].min()
    if vmax is None:
        vmax = data['density'].max()
    if level_width is None:
        level_width = (vmax - vmin) / float(n_levels) / 5.

    # Set up color transfer function
    transfer = ColorTransferFunction((vmin, vmax))

    # Set up the camera parameters
    if center is None:
        center = [cube.shape[0] / 2. * spectral_factor,
                  cube.shape[1] / 2.,
                  cube.shape[2] / 2.]
    else:
        lon, lat, spectral = center
        ilon, ilat, ispectral = cube.wcs.wcs_world2pix(lon, lat, spectral, 0)
        center = [ispectral, ilat, ilon]

    looking_direction = np.array([0.0, 1.0, 0.0])
    width = max(cube.shape[0] * spectral_factor, cube.shape[1], cube.shape[2]) / float(zoom)

    camera = pf.h.camera(center, looking_direction, width, size, transfer,
                         fields=['density'])

    transfer.add_layers(n_levels, level_width, colormap='RdBu_r')

    if fly_around is None:
        snapshot = camera.snapshot()
        write_bitmap(snapshot, filename, transpose=True)
    else:
        for i, snapshot in enumerate(camera.rotation(2. * np.pi, fly_around, clip_ratio=8.0)):
            camera.snapshot()
            write_bitmap(snapshot, filename.replace('.png', '_{0:04d}.png'.format(i)), transpose=True)
