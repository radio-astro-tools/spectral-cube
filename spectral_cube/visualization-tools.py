import os
import numpy as np
import aplpy
from astropy.utils.console import ProgressBar

def make_rgb_movie(cube, prefix, v1, v2, vmin, vmax, ffmpeg_cmd='ffmpeg'):
    """
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
   
    # Create the WCS template
    F = aplpy.FITSFigure(cube[0].hdu)
 
    # determine pixel range
    p1 = cube.closest_spectral_channel(v1)
    p2 = cube.closest_spectral_channel(v2)
 
    for jj,ii in enumerate(ProgressBar(range(p1, p2-1))):
        rgb = np.array([cube[ii+2], cube[ii+1], cube[ii]]).T.swapaxes(0,1)
 
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
        if os.path.exists('{prefix}{0:04d}.png'.format(jj,prefix=prefix)):
            os.remove('{prefix}{0:04d}.png'.format(jj,prefix=prefix))
        os.link('{2}_v{0}to{1}.png'.format(v1_, v2_, prefix),
                '{prefix}{0:04d}.png'.format(jj, prefix=prefix))
 
    os.system('{ffmpeg} -y -i {prefix}%04d.png -c:v libx264 -pix_fmt yuv420p -vf '
              '"scale=1024:768,setpts=10*PTS" -r 10'
              ' {prefix}_RGB_movie.mp4'.format(prefix=prefix, ffmpeg=ffmpeg_cmd))
