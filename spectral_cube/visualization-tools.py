import os
import numpy as np
from astropy.utils.console import ProgressBar

def check_ffmpeg(ffmpeg_cmd):
    returncode = os.system(f'{ffmpeg_cmd} > /dev/null 2&> /dev/null')
    if returncode not in (0, 256):
        raise OSError(f"{ffmpeg_cmd} not found in the executable path."
                      f"  Return code {returncode}")


def make_rgb_movie(cube, prefix, v1, v2, vmin, vmax, ffmpeg_cmd='ffmpeg'):
    """
    Make a velocity movie with red, green, and blue corresponding to
    neighboring velocity channels

    Parameters
    ----------
    cube : SpectralCube
        The cube to visualize
    prefix : str
        A prefix to prepend to the output png and movie.  For example,
        it could be rgb/sourcename_speciesname
    v1 : Quantity
        A value in spectral-axis equivalent units specifying
        the starting velocity / frequency / wavelength
    v2 : Quantity
        A value in spectral-axis equivalent units specifying
        the ending velocity / frequency / wavelength
    vmin : float
        Minimum value to display
    vmax : float
        Maximum value to display
    ffmpeg_cmd : str
        The system command for ffmpeg.  Required to make a movie
    """
    import aplpy

    check_ffmpeg(ffmpeg_cmd)

    # Create the WCS template
    F = aplpy.FITSFigure(cube[0].hdu)
    F.show_grayscale()

    # determine pixel range
    p1 = cube.closest_spectral_channel(v1)
    p2 = cube.closest_spectral_channel(v2)

    for jj, ii in enumerate(ProgressBar(range(p1, p2-1))):
        rgb = np.array([cube[ii+2], cube[ii+1], cube[ii]]).T.swapaxes(0, 1)

        # in case you manually set min/max
        rgb[rgb > vmax] = 1
        rgb[rgb < vmin] = 0

        # this is the unsupported little bit...
        F._ax1.clear()
        F._ax1.imshow((rgb-vmin)/(vmax-vmin), extent=F._extent)

        v1_ = int(np.round(cube.spectral_axis[ii].value))
        v2_ = int(np.round(cube.spectral_axis[ii+2].value))

        # then write out the files
        F.save('{2}_v{0}to{1}.png'.format(v1_, v2_, prefix))
        # make a sorted version for use with ffmpeg
        if os.path.exists('{prefix}{0:04d}.png'.format(jj, prefix=prefix)):
            os.remove('{prefix}{0:04d}.png'.format(jj, prefix=prefix))
        os.link('{2}_v{0}to{1}.png'.format(v1_, v2_, prefix),
                '{prefix}{0:04d}.png'.format(jj, prefix=prefix))

    os.system('{ffmpeg} -r 10 -y -i {prefix}%04d.png -c:v '
              'libx264 -pix_fmt yuv420p -vf '
              '"scale=1024:768" -r 10' # "scale=1024:768,setpts=10*PTS"
              ' {prefix}_RGB_movie.mp4'.format(prefix=prefix,
                                               ffmpeg=ffmpeg_cmd))

def make_multispecies_rgb(cube_r, cube_g, cube_b, prefix, v1, v2, vmin, vmax,
                          ffmpeg_cmd='ffmpeg'):
    """

    Parameters
    ----------
    cube_r, cube_g, cube_b : SpectralCube
        The three cubes to visualize.  They should have identical spatial and
        spectral dimensions.
    prefix : str
        A prefix to prepend to the output png and movie.  For example,
        it could be rgb/sourcename_speciesname
    v1 : Quantity
        A value in spectral-axis equivalent units specifying
        the starting velocity / frequency / wavelength
    v2 : Quantity
        A value in spectral-axis equivalent units specifying
        the ending velocity / frequency / wavelength
    vmin : float
        Minimum value to display (constant for all 3 colors)
    vmax : float
        Maximum value to display (constant for all 3 colors)
    ffmpeg_cmd : str
        The system command for ffmpeg.  Required to make a movie
    """
    import aplpy

    check_ffmpeg(ffmpeg_cmd)

    # Create the WCS template
    F = aplpy.FITSFigure(cube_r[0].hdu)
    F.show_grayscale()

    assert cube_r.shape == cube_b.shape
    assert cube_g.shape == cube_b.shape

    # determine pixel range
    p1 = cube_r.closest_spectral_channel(v1)
    p2 = cube_r.closest_spectral_channel(v2)

    for jj, ii in enumerate(ProgressBar(range(p1, p2+1))):
        rgb = np.array([cube_r[ii].value,
                        cube_g[ii].value,
                        cube_b[ii].value
                       ]).T.swapaxes(0, 1)

        # in case you manually set min/max
        rgb[rgb > vmax] = 1
        rgb[rgb < vmin] = 0

        # this is the unsupported little bit...
        F._ax1.clear()
        F._ax1.imshow((rgb-vmin)/(vmax-vmin), extent=F._extent)

        v1_ = int(np.round(cube_r.spectral_axis[ii].value))

        # then write out the files
        F.refresh()
        F.save('{1}_v{0}.png'.format(v1_, prefix))
        # make a sorted version for use with ffmpeg
        if os.path.exists('{prefix}{0:04d}.png'.format(jj, prefix=prefix)):
            os.remove('{prefix}{0:04d}.png'.format(jj, prefix=prefix))
        os.link('{1}_v{0}.png'.format(v1_, prefix),
                '{prefix}{0:04d}.png'.format(jj, prefix=prefix))

    os.system('{ffmpeg} -r 10 -y -i {prefix}%04d.png'
              ' -c:v libx264 -pix_fmt yuv420p -vf '
              '"scale=1024:768" -r 10' # "scale=1024:768,setpts=10*PTS"
              ' {prefix}_RGB_movie.mp4'.format(prefix=prefix,
                                               ffmpeg=ffmpeg_cmd))
