# The read and write methods for SpectralCube, StokesSpectralCube, and
# LowerDimensionalObject are defined in this file and then added to the classes
# using UnifiedReadWriteMethod. This makes it possible to dynamically add the
# available formats to the read/write docstrings. For more information about
# the unified I/O framework from Astropy which is used to implement this, see
# http://docs.astropy.org/en/stable/io/unified.html

from __future__ import print_function, absolute_import, division

from pathlib import PosixPath
import warnings

from astropy.io import registry

from ..utils import StokesWarning

__doctest_skip__ = ['SpectralCubeRead',
                    'SpectralCubeWrite',
                    'StokesSpectralCubeRead',
                    'StokesSpectralCubeWrite',
                    'LowerDimensionalObjectWrite']

DOCSTRING_READ_TEMPLATE = """
Read and parse a dataset and return as a {clsname}

This allows easily reading a dataset in several supported data
formats using syntax such as::

    >>> from spectral_cube import {clsname}
    >>> cube1 = {clsname}.read('cube.fits', format='fits')
    >>> cube2 = {clsname}.read('cube.image', format='casa')

{notes}

Get help on the available readers for {clsname} using the``help()`` method::

    >>> {clsname}.read.help()  # Get help reading {clsname} and list supported formats
    >>> {clsname}.read.help('fits')  # Get detailed help on {clsname} FITS reader
    >>> {clsname}.read.list_formats()  # Print list of available formats

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
cube : `{clsname}`
    {clsname} corresponding to dataset

Notes
-----
"""

DOCSTRING_WRITE_TEMPLATE = """
Write this {clsname} object out in the specified format.

This allows easily writing a dataset in many supported data formats
using syntax such as::

    >>> data.write('data.fits', format='fits')

Get help on the available writers for {clsname} using the``help()`` method::

    >>> {clsname}.write.help()  # Get help writing {clsname} and list supported formats
    >>> {clsname}.write.help('fits')  # Get detailed help on {clsname} FITS writer
    >>> {clsname}.write.list_formats()  # Print list of available formats

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


# Due to a bug in the astropy I/O infrastructure which causes an exception
# for directories (which we need for .image), we need to wrap the filenames in
# a custom string so that astropy doesn't try and call get_readable_fileobj on
# them.
class StringWrapper:
    def __init__(self, value):
        self.value = value


class SpectralCubeRead(registry.UnifiedReadWrite):

    __doc__ = DOCSTRING_READ_TEMPLATE.format(clsname='SpectralCube',
                                             notes="If the file contains Stokes axes, they will automatically be dropped. If "
                                                    "you want to read in all Stokes informtion, use "
                                                    ":meth:`~spectral_cube.StokesSpectralCube.read` instead.")

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'read')

    def __call__(self, filename, *args, **kwargs):
        from ..spectral_cube import BaseSpectralCube
        if isinstance(filename, PosixPath):
            filename = str(filename)
        kwargs['target_cls'] = BaseSpectralCube
        try:
            return registry.read(BaseSpectralCube, filename, *args, **kwargs)
        except IsADirectoryError:  # See note above StringWrapper
            return registry.read(BaseSpectralCube, StringWrapper(filename), *args, **kwargs)


class SpectralCubeWrite(registry.UnifiedReadWrite):

    __doc__ = DOCSTRING_WRITE_TEMPLATE.format(clsname='SpectralCube')

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'write')

    def __call__(self, *args, serialize_method=None, **kwargs):
        registry.write(self._instance, *args, **kwargs)


class StokesSpectralCubeRead(registry.UnifiedReadWrite):

    __doc__ = DOCSTRING_READ_TEMPLATE.format(clsname='StokesSpectralCube',
                                             notes="If the file contains Stokes axes, they will be read in. If you are only "
                                                   "interested in the unpolarized emission (I), you can use "
                                                   ":meth:`~spectral_cube.SpectralCube.read` instead.")

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'read')

    def __call__(self, filename, *args, **kwargs):
        from ..stokes_spectral_cube import StokesSpectralCube
        if isinstance(filename, PosixPath):
            filename = str(filename)
        kwargs['target_cls'] = StokesSpectralCube
        try:
            return registry.read(StokesSpectralCube, filename, *args, **kwargs)
        except IsADirectoryError:  # See note above StringWrapper
            return registry.read(StokesSpectralCube, StringWrapper(filename), *args, **kwargs)


class StokesSpectralCubeWrite(registry.UnifiedReadWrite):

    __doc__ = DOCSTRING_WRITE_TEMPLATE.format(clsname='StokesSpectralCube')

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'write')

    def __call__(self, *args, serialize_method=None, **kwargs):
        registry.write(self._instance, *args, **kwargs)


class LowerDimensionalObjectWrite(registry.UnifiedReadWrite):

    __doc__ = DOCSTRING_WRITE_TEMPLATE.format(clsname='LowerDimensionalObject')

    def __init__(self, instance, cls):
        super().__init__(instance, cls, 'write')

    def __call__(self, *args, serialize_method=None, **kwargs):
        registry.write(self._instance, *args, **kwargs)


def normalize_cube_stokes(cube, target_cls=None):

    from ..spectral_cube import BaseSpectralCube
    from ..stokes_spectral_cube import StokesSpectralCube

    if target_cls is BaseSpectralCube and isinstance(cube, StokesSpectralCube):
        if hasattr(cube, 'I'):
            warnings.warn("Cube is a Stokes cube, "
                          "returning spectral cube for I component",
                          StokesWarning)
            return cube.I
        else:
            raise ValueError("Spectral cube is a Stokes cube that "
                            "does not have an I component")
    elif target_cls is StokesSpectralCube and isinstance(cube, BaseSpectralCube):
        return StokesSpectralCube({'I': cube})
    else:
        return cube
