from __future__ import print_function, absolute_import, division

import warnings

from astropy.io import registry

__doctest_skip__ = ['SpectralCubeRead', 'SpectralCubeWrite']


class SpectralCubeRead(registry.UnifiedReadWrite):
    """
    Read and parse a dataset and return as a SpectralCube

    This function provides the SpectralCube interface to the astropy unified I/O
    layer. This allows easily reading a dataset in several supported data
    formats using syntax such as::

      >>> from spectral_cube import SpectralCube
      >>> cube1 = SpectralCube.read('cube.fits', format='fits')
      >>> cube2 = SpectralCube.read('cube.image', format='casa')

    If the file contains Stokes axes, they will automatically be dropped. If
    you want to read in all Stokes informtion, use
    :meth:`~spectral_cube.StokesSpectralCube.read` instead.

    Get help on the available readers for ``SpectralCube`` using the``help()`` method::

      >>> SpectralCube.read.help()  # Get help reading SpectralCube and list supported formats
      >>> SpectralCube.read.help('fits')  # Get detailed help on SpectralCube FITS reader
      >>> SpectralCube.read.list_formats()  # Print list of available formats

    See also: http://docs.astropy.org/en/stable/io/unified.html

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments passed through to data reader. If supplied the
        first argument is typically the input filename.
    format : str
        File format specifier.
    **kwargs : dict, optional
        Keyword arguments passed through to data reader.

    Returns
    -------
    cube : `SpectralCube`
        SpectralCube corresponding to dataset

    Notes
    -----
    """

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'read')

    def __call__(self, *args, **kwargs):
        from .. import StokesSpectralCube
        from ..utils import StokesWarning
        cube = registry.read(self._cls, *args, **kwargs)
        if isinstance(cube, StokesSpectralCube):
            if hasattr(cube, 'I'):
                warnings.warn("Cube is a Stokes cube, "
                              "returning spectral cube for I component",
                              StokesWarning)
                return cube.I
            else:
                raise ValueError("Spectral cube is a Stokes cube that "
                                "does not have an I component")
        else:
            return cube


class SpectralCubeWrite(registry.UnifiedReadWrite):
    """
    Write this SpectralCube object out in the specified format.

    This function provides the SpectralCube interface to the astropy unified
    I/O layer.  This allows easily writing a spectral cube in many supported
    data formats using syntax such as::

      >>> cube.write('cube.fits', format='fits')

    Get help on the available writers for ``SpectralCube`` using the``help()`` method::

      >>> SpectralCube.write.help()  # Get help writing SpectralCube and list supported formats
      >>> SpectralCube.write.help('fits')  # Get detailed help on SpectralCube FITS writer
      >>> SpectralCube.write.list_formats()  # Print list of available formats

    See also: http://docs.astropy.org/en/stable/io/unified.html

    Parameters
    ----------
    *args : tuple, optional
        Positional arguments passed through to data writer. If supplied the
        first argument is the output filename.
    format : str
        File format specifier.
    **kwargs : dict, optional
        Keyword arguments passed through to data writer.

    Notes
    -----
    """
    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'write')

    def __call__(self, *args, serialize_method=None, **kwargs):
        registry.write(self._instance, *args, **kwargs)


def read(filename, format=None, hdu=None, **kwargs):
    """
    Read a file into a :class:`SpectralCube` or :class:`StokesSpectralCube`
    instance.

    Parameters
    ----------
    filename : str or HDU
        File to read
    format : str, optional
        File format.
    hdu : int or str
        For FITS files, the HDU to read in (can be the ID or name of an
        HDU).
    kwargs : dict
        If the format is 'fits', the kwargs are passed to
        :func:`~astropy.io.fits.open`.

    Returns
    -------
    cube : :class:`SpectralCube` or :class:`StokesSpectralCube`
        The spectral cube read in
    """

    if format is None:
        format = determine_format(filename)

    if format == 'fits':
        from .fits import load_fits_cube
        return load_fits_cube(filename, hdu=hdu, **kwargs)
    elif format == 'casa_image':
        from .casa_image import load_casa_image
        return load_casa_image(filename)
    elif format in ('class_lmv','lmv'):
        from .class_lmv import load_lmv_cube
        return load_lmv_cube(filename)
    else:
        raise ValueError("Format {0} not implemented. Supported formats are "
                         "'fits', 'casa_image', and 'lmv'.".format(format))


def write(filename, cube, overwrite=False, format=None):
    """
    Write :class:`SpectralCube` or :class:`StokesSpectralCube` to a file.

    Parameters
    ----------
    filename : str
        Name of the output file
    cube : :class:`SpectralCube` or :class:`StokesSpectralCube`
        The spectral cube to write out
    overwrite : bool, optional
        Whether to overwrite the output file
    format : str, optional
        File format.
    """

    if format is None:
        format = determine_format(filename)

    if format == 'fits':
        from .fits import write_fits_cube
        write_fits_cube(filename, cube, overwrite=overwrite)
    else:
        raise ValueError("Format {0} not implemented. The only supported format is 'fits'".format(format))


def determine_format(input):

    from .fits import is_fits
    from .casa_image import is_casa_image
    from .class_lmv import is_lmv

    if is_fits(input):
        return 'fits'
    elif is_casa_image(input):
        return 'casa_image'
    elif is_lmv(input):
        return 'lmv'
    else:
        raise ValueError("Could not determine format - use the `format=` "
                         "parameter to explicitly set the format")
