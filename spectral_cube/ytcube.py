import numpy as np
from astropy.utils.console import ProgressBar
import os
from astropy import log
import subprocess

try:
    import yt
    from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
    ytOK = True
except ImportError:
    ytOK = False

class ytCube(object):
    """ Light wrapper of a yt object with ability to translate yt<->wcs
    coordinates """

    def __init__(self, cube, dataset, spectral_factor=1.0):
        self.cube = cube
        self.wcs = cube.wcs
        self.dataset = dataset
        self.spectral_factor = spectral_factor


    def world2yt(self, world_coord, first_index=0):
        """
        Convert a position in world coordinates to the coordinates used by a
        yt dataset that has been generated using the ``to_yt`` method.

        Parameters
        ----------
        world_coord: `astropy.wcs.WCS.wcs_world2pix`-valid input
            The world coordinates
        first_index: 0 or 1
            The first index of the data.  In python and yt, this should be
            zero, but for the FITS coordinates, use 1
        """
        yt_coord = self.wcs.wcs_world2pix([world_coord], first_index)[0]
        yt_coord[2] = (yt_coord[2] - 0.5)*self.spectral_factor+0.5
        return yt_coord

    def yt2world(self, yt_coord, first_index=0):
        """
        Convert a position in yt's coordinates to world coordinates from a
        yt dataset that has been generated using the ``to_yt`` method.

        Parameters
        ----------
        world_coord: `astropy.wcs.WCS.wcs_pix2world`-valid input
            The yt pixel coordinates to convert back to world coordinates
        first_index: 0 or 1
            The first index of the data.  In python and yt, this should be
            zero, but for the FITS coordinates, use 1
        """
        yt_coord = np.array(yt_coord) # stripping off units
        yt_coord[2] = (yt_coord[2] - 0.5)/self.spectral_factor+0.5
        world_coord = self.wcs.wcs_pix2world([yt_coord], first_index)[0]
        return world_coord


    def quick_render_movie(self, outdir, size=256, nframes=30,
                           camera_angle=[0,0,1], north_vector = [1, 0, 0],
                           rot_vector=[1,0,0],
                           colormap='doom',
                           cmap_range='auto',
                           log=False,
                           rescale=True):
        """
        Create a movie rotating the cube 360 degrees from
        PP -> PV -> PP -> PV -> PP

        Parameters
        ----------
        outdir: str

        """
        if not ytOK:
            raise IOError("yt could not be imported.  Cube renderings are not possible.")

        scale = np.max(self.cube.shape)

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif not os.path.isdir(outdir):
            raise OSError("Output directory {0} exists and is not a directory.".format(outdir))

        if cmap_range == 'auto':
            upper = self.cube.max().value
            lower = self.cube.std().value * 3
            cmap_range = [lower,upper]

        tfh = TransferFunctionHelper(self.dataset)
        tfh.set_field('flux')
        tfh.set_bounds(bounds=cmap_range)
        tfh.set_log(log)
        tfh.build_transfer_function()
        tfh.tf.map_to_colormap(cmap_range[0], cmap_range[1], colormap=colormap)

        center = self.dataset.domain_dimensions / 2.
        cam = self.dataset.h.camera(center, camera_angle, scale, size, tfh.tf,
                                    north_vector=north_vector, fields='flux')

        im  = cam.snapshot()
        images = [im]

        pb = ProgressBar(nframes)
        for ii,im in enumerate(cam.rotation(2 * np.pi, nframes,
                                            rot_vector=rot_vector)):
            images.append(im)
            im.write_png(os.path.join(outdir,"%04i.png" % (ii)), rescale=False)
            pb.update(ii)

        if rescale:
            _rescale_images(images, outdir)

        pipe = _make_movie(outdir)
        
        return images

def _rescale_images(images, prefix):
    """
    Save a sequence of images, at a common scaling
    Reduces flickering
    """
 
    cmax = max(np.percentile(i[:, :, :3].sum(axis=2), 99.5) for i in images)
    amax = max(np.percentile(i[:, :, 3], 95) for i in images)
 
    for i, image in enumerate(images):
        image = image.rescale(cmax=cmax, amax=amax).swapaxes(0,1)
        image.write_png(os.path.join(prefix, "%04i.png" % (i)), rescale=False)

def _make_movie(moviepath, overwrite=True):
    """
    Use ffmpeg to generate a movie from the image series
    """

    outpath = os.path.join(moviepath,'out.mp4')

    if os.path.exists(outpath) and overwrite:
        command = ['ffmpeg', '-y', '-r','5','-i',
                   os.path.join(moviepath,'%04d.png'),
                   '-c:v','libx264','-r','30','-pix_fmt', 'yuv420p',
                   outpath]
    elif os.path.exists(outpath):
        log.info("File {0} exists - skipping".format(outpath))
    else:
        command = ['ffmpeg', '-r', '5', '-i',
                   os.path.join(moviepath,'%04d.png'),
                   '-c:v','libx264','-r','30','-pix_fmt', 'yuv420p',
                   outpath]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, close_fds=True)

    pipe.wait()

    return pipe
