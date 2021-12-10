.. doctest-skip-all
.. The example below isn't meant to work

============
Reprojection
============

Spectral-cube has several tools to enable reprojection of cubes onto different spatial and spectral grids.

Prior to reprojecting data, in order to minimize resampling artifacts, it is a
good idea to :doc:`smooth <smoothing>` the data first.  A worked example of spatial
and spectral smoothing is given on the `reprojection tutorial
<https://github.com/radio-astro-tools/tutorials/blob/master/SpectralCubeReprojectExample.ipynb>`_.


Spatial Reprojection
^^^^^^^^^^^^^^^^^^^^

To reproject a cube onto a different spatial world coordinate system, use the
:meth:`~spectral_cube.SpectralCube.reproject` function.  The function requires
a target header as an input.  You might generate this header by grabbing it
from another FITS file, for example, from another `SpectralCube`::

    from spectral_cube import SpectralCube
    
    cube = SpectralCube.read('/some_path/some_file.fits')
    other_cube = SpectralCube.read('/some_path/other_file.fits')

    reprojected_cube = cube.reproject(other_cube.header)

Instead, the target header can be generated it with a helper tool (e.g., the
`find_optimal_celestial_wcs
<https://reproject.readthedocs.io/en/stable/mosaicking.html#computing-an-optimal-wcs>`_
function in the `reproject <https://reproject.readthedocs.io/>`_ package), or
by manually editing a FITS header.

The spatial reprojection tool uses reproject_ under the hood and defaults to
using a bilinear interpolation scheme, though this is configurable.
Interpolation onto a differently-spaced grid, after appropriate :doc:`smoothing <smoothing>`, can
be used to rebin or decimate the data.

A simple example for rebinning, assuming no smoothing is needed (appropriate for when the data are oversampled)::


    from spectral_cube import SpectralCube

    cube = SpectralCube.read('/some_path/some_file.fits')

    # create a target header to reproject to by making the pixel size 2 times larger
    target_header = cube.wcs.celestial[::2, ::2].to_header()
    target_header['NAXIS1'] = cube.shape[2] / 2
    target_header['NAXIS2'] = cube.shape[1] / 2

    downsampled_cube = cube.reproject(target_header)

Reprojection for 2D images uses the same syntax with a `~spectral_cube.Projection` or `~spectral_cube.Slice` object. For example, to match the spatial grid of a 2D image to that of a cube::

        from spectral_cube import SpectralCube, Projection
        from astropy.io import fits
    
        cube = SpectralCube.read('/some_path/some_file.fits')
        image = Projection.from_hdu(fits.open('/some_path/twod_image.fits')[0])
        
        cube_header_spatial = cube.wcs.celestial.to_header()
        reprojected_image = image.reproject(cube_header_spatial)


Spectral Reprojection
^^^^^^^^^^^^^^^^^^^^^

Spectral reprojection behaves similar to spatial reprojection.
The :meth:`~spectral_cube.SpectralCube.spectral_interpolate` function
allows interpolation of the data onto a new spectral grid.
Unlike spatial reprojection, though, the expected input is a list
of pixel coordinates.  See the example in the :ref:`Spectral Smoothing <Spectral-Smoothing>` section of
the :doc:`smoothing` document.
