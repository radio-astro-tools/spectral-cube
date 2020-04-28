from __future__ import print_function, absolute_import, division

import six
import os
import subprocess
import numpy as np
import time
from astropy.utils.console import ProgressBar
from astropy import log
import warnings

__all__ = ['ytCube']


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
                           camera_angle=(0,0,1),
                           north_vector=(0,1,0),
                           rot_vector=None,
                           zoom=None,
                           colormap='doom',
                           cmap_range='auto',
                           transfer_function='auto',
                           start_index=0,
                           image_prefix="",
                           output_filename='out.mp4',
                           log_scale=False, run_ffmpeg=True,
                           rescale=True, sigma_clip=None):
        """
        Create a movie rotating the cube 360 degrees from
        PP -> PV -> PP -> PV -> PP

        Parameters
        ----------
        outdir: str
            The output directory in which the individual image frames and the
            resulting output mp4 file should be stored
        size: int
            The size of the individual output frame in pixels (i.e., size=256
            will result in a 256x256 image)
        nframes: int
            The number of frames in the resulting movie
        camera_angle: 3-tuple
            The initial angle of the camera
        north_vector: 3-tuple
            The vector of 'north' in the data cube. Default is the "y" direction
        rot_vector: 3-tuple
            The vector around which the camera will be rotated. Default: None,
            which means it will be set to the north_vector value.
        zoom : float
            Change the width of the FOV of the camera. Default: None, which 
            does no zooming. 
        colormap: str
            A valid colormap.  See `yt.show_colormaps`
        cmap_range : 2-tuple or "auto"
            If a 2-tuple of floats, this will be the (vmin, vmax) for the colorbar.
            Otherise, "auto" sets the values from the data.
        transfer_function: 'auto' or `yt.visualization.volume_rendering.TransferFunction`
            Either 'auto' to use the colormap specified, or a valid
            TransferFunction instance
        start_index : int
            The number of the first image to save
        image_prefix : str
            A string to prepend to the image name for each image that is output
        output_filename : str
            The movie file name to output.  The suffix may affect the file type
            created.  Defaults to 'out.mp4'.  Will be placed in ``outdir``
        log_scale : bool
            Should the colormap be log scaled?
        run_ffmpeg : bool
            If True, ffmpeg will be used to make a movie out of the images.
            Default: True
        rescale: bool
            If True, the images will be rescaled to have a common 95th
            percentile brightness, which can help reduce flickering from having
            a single bright pixel in some projections
        sigma_clip: float, optional
            Image values greater than this number times the standard deviation
            plus the mean of the image will be clipped before saving. Useful
            for enhancing images as it gets rid of rare high pixel values.
            Default: None
        """
        try:
            import yt
        except ImportError:
            raise ImportError("yt could not be imported. Cube renderings are not possible.")
        else:
            from packaging import version
            if version.parse(yt.__version__) < version.parse("3.5.0"):
                raise RuntimeError("Only yt versions >= 3.5.0 are supported. Please upgrade yt.")

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        elif not os.path.isdir(outdir):
            raise OSError("Output directory {0} exists and is not a directory.".format(outdir))

        if rot_vector is None:
            rot_vector = north_vector

        if cmap_range == 'auto':
            upper = self.cube.max().value
            lower = self.cube.std().value * 3
            cmap_range = [lower,upper]

        data_source = self.dataset.all_data()
        sc = yt.create_scene(data_source, field='flux')
        source = sc[0]

        if transfer_function == 'auto':
            tfh = self.auto_transfer_function(cmap_range, log=log_scale)
            tfh.tf.map_to_colormap(cmap_range[0], cmap_range[1], colormap=colormap)
            source.tfh = tfh
        else:
            tf = transfer_function
            source.tfh.tf = tf
            source.tfh.bounds = cmap_range

        cam = sc.camera
        cam.set_focus(data_source.get_field_parameter("center"))
        cam.set_resolution(size)
        cam.switch_view(normal_vector=camera_angle, north_vector=north_vector)
        if zoom is not None:
            cam.zoom(zoom)

        im = sc.render()
        images = [im]

        pb = ProgressBar(nframes)
        for ii in cam.iter_rotate(2*np.pi, nframes, rot_vector=rot_vector):
            im = sc.render()
            images.append(im)
            outfile = os.path.join(outdir, "%s%04i.png" % (image_prefix,
                                                           ii+start_index))
            im.write_png(outfile, sigma_clip=sigma_clip, rescale=False)
            pb.update(ii+1)
        log.info("Rendering complete in {0}s".format(time.time() - pb._start_time))

        if rescale:
            _rescale_images(images, os.path.join(outdir, image_prefix))

        if run_ffmpeg:
            pipe = _make_movie(outdir, prefix=image_prefix,
                               filename=output_filename)

        return images

    def auto_transfer_function(self, cmap_range, log=False):
        from yt.visualization.volume_rendering.transfer_function_helper import \
            TransferFunctionHelper
        tfh = TransferFunctionHelper(self.dataset)
        tfh.set_field('flux')
        tfh.set_bounds(bounds=cmap_range)
        tfh.set_log(log)
        tfh.build_transfer_function()
        return tfh

    def quick_isocontour(self, level='3 sigma', title='', description='',
                         color_map='hot', color_log=False,
                         export_to='sketchfab', filename=None,
                         **kwargs):
        """
        Export isocontours to sketchfab

        Requires that you have an account on https://sketchfab.com and are
        logged in

        Parameters
        ----------
        level: str or float
            The level of the isocontours to create.  Can be specified as
            n-sigma with strings like '3.3 sigma' or '2 sigma' (there must be a
            space between the number and the word)
        title: str
            A title for the uploaded figure
        description: str
            A short description for the uploaded figure
        color_map: str
            Any valid colormap.  See `yt.show_colormaps`
        color_log: bool
            Whether the colormap should be log scaled.  With the default
            parameters, this has no effect.
        export_to: 'sketchfab', 'obj', 'ply'
            You can export to sketchfab, to a .obj file (and accompanying .mtl
            file), or a .ply file.  The latter two require ``filename``
            specification
        filename: None or str
            Optional - prefix for output filenames if ``export_to`` is 'obj',
            or the full filename when ``export_to`` is 'ply'.  Ignored for
            'sketchfab'
        kwargs: dict
            Keyword arguments are passed to the appropriate yt function

        Returns
        -------
        The result of the `yt.surface.export_sketchfab` function
        """
        if isinstance(level, six.string_types):
            sigma = self.cube.std().value
            level = float(level.split()[0]) * sigma

        self.dataset.periodicity = (True,True,True)
        surface = self.dataset.surface(self.dataset.all_data(),
                                       "flux",
                                       level)
        if export_to == 'sketchfab':
            if filename is not None:
                warnings.warn("sketchfab export does not expect a filename entry")
            return surface.export_sketchfab(title=title,
                                            description=description,
                                            color_map=color_map,
                                            color_log=color_log, **kwargs)
        elif export_to == 'obj':
            if filename is None:
                raise ValueError("If export_to is not 'sketchfab',"
                                 " a filename must be specified")

            surface.export_obj(filename, color_field='ones',
                               color_map=color_map, color_log=color_log,
                               **kwargs)
        elif export_to == 'ply':
            if filename is None:
                raise ValueError("If export_to is not 'sketchfab',"
                                 " a filename must be specified")

            surface.export_ply(filename, color_field='ones',
                               color_map=color_map, color_log=color_log,
                               **kwargs)
        else:
            raise ValueError("export_to must be one of sketchfab,obj,ply")


def _rescale_images(images, prefix):
    """
    Save a sequence of images, at a common scaling
    Reduces flickering
    """

    cmax = max(np.percentile(i[:, :, :3].sum(axis=2), 99.5) for i in images)
    amax = max(np.percentile(i[:, :, 3], 95) for i in images)

    for i, image in enumerate(images):
        image = image.rescale(cmax=cmax, amax=amax).swapaxes(0,1)
        image.write_png("%s%04i.png" % (prefix, i), rescale=False)


def _make_movie(moviepath, prefix="", filename='out.mp4', overwrite=True):
    """
    Use ffmpeg to generate a movie from the image series
    """

    outpath = os.path.join(moviepath, filename)

    if os.path.exists(outpath) and overwrite:
        command = ['ffmpeg', '-y', '-r','5','-i',
                   os.path.join(moviepath,prefix+'%04d.png'),
                   '-r','30','-pix_fmt', 'yuv420p',
                   outpath]
    elif os.path.exists(outpath):
        log.info("File {0} exists - skipping".format(outpath))
    else:
        command = ['ffmpeg', '-r', '5', '-i',
                   os.path.join(moviepath,prefix+'%04d.png'),
                   '-r','30','-pix_fmt', 'yuv420p',
                   outpath]

    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, close_fds=True)

    pipe.wait()

    return pipe
