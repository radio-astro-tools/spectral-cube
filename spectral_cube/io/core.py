from __future__ import print_function, absolute_import, division

from pathlib import PosixPath
import warnings

from astropy.io import registry

__doctest_skip__ = ['SpectralCubeRead',
                    'SpectralCubeWrite',
                    'StokesSpectralCubeRead',
                    'StokesSpectralCubeWrite',
                    'LowerDimensionalObjectWrite']


class SpectralCubeRead(registry.UnifiedReadWrite):
    """
    Read and parse a dataset and return as a SpectralCube

    This allows easily reading a dataset in several supported data
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

    def __call__(self, filename, *args, **kwargs):
        from ..spectral_cube import BaseSpectralCube
        if isinstance(filename, PosixPath):
            filename = str(filename)
        kwargs['target_cls'] = BaseSpectralCube
        return registry.read(BaseSpectralCube, filename, *args, **kwargs)


class SpectralCubeWrite(registry.UnifiedReadWrite):
    """
    Write this SpectralCube object out in the specified format.

    This allows easily writing a spectral cube in many supported data formats
    using syntax such as::

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


class StokesSpectralCubeRead(registry.UnifiedReadWrite):
    """
    Read and parse a dataset and return as a StokesSpectralCube

    This allows easily reading a dataset in several supported data formats
    using syntax such as::

      >>> from spectral_cube import StokesSpectralCube
      >>> cube1 = StokesSpectralCube.read('cube.fits', format='fits')
      >>> cube2 = StokesSpectralCube.read('cube.image', format='casa')

    If the file contains Stokes axes, they will be read in. If you are only
    interested in the unpolarized emission (I), you can use
    :meth:`~spectral_cube.SpectralCube.read` instead.

    Get help on the available readers for ``StokesSpectralCube`` using the``help()`` method::

      >>> StokesSpectralCube.read.help()  # Get help reading StokesSpectralCube and list supported formats
      >>> StokesSpectralCube.read.help('fits')  # Get detailed help on StokesSpectralCube FITS reader
      >>> StokesSpectralCube.read.list_formats()  # Print list of available formats

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
    cube : `StokesSpectralCube`
        StokesSpectralCube corresponding to dataset

    Notes
    -----
    """

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'read')

    def __call__(self, filename, *args, **kwargs):
        from ..stokes_spectral_cube import StokesSpectralCube
        if isinstance(filename, PosixPath):
            filename = str(filename)
        kwargs['target_cls'] = StokesSpectralCube
        return registry.read(StokesSpectralCube, filename, *args, **kwargs)


class StokesSpectralCubeWrite(registry.UnifiedReadWrite):
    """
    Write this StokesSpectralCube object out in the specified format.

    This allows easily writing a spectral cube in many supported data formats
    using syntax such as::

      >>> cube.write('cube.fits', format='fits')

    Get help on the available writers for ``StokesSpectralCube`` using the``help()`` method::

      >>> StokesSpectralCube.write.help()  # Get help writing StokesSpectralCube and list supported formats
      >>> StokesSpectralCube.write.help('fits')  # Get detailed help on StokesSpectralCube FITS writer
      >>> StokesSpectralCube.write.list_formats()  # Print list of available formats

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


class LowerDimensionalObjectWrite(registry.UnifiedReadWrite):
    """
    Write this object out in the specified format.

    This allows easily writing a data object in many supported data formats
    using syntax such as::

      >>> data.write('data.fits', format='fits')

    Get help on the available writers using the``help()`` method, e.g.::

      >>> LowerDimensionalObject.write.help()  # Get help writing LowerDimensionalObject and list supported formats
      >>> LowerDimensionalObject.write.help('fits')  # Get detailed help on LowerDimensionalObject FITS writer
      >>> LowerDimensionalObject.write.list_formats()  # Print list of available formats

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
