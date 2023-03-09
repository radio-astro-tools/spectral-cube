# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ._astropy_init import __version__, test

from .spectral_cube import (SpectralCube, VaryingResolutionSpectralCube)
from .dask_spectral_cube import (DaskSpectralCube, DaskVaryingResolutionSpectralCube)
from .stokes_spectral_cube import StokesSpectralCube
from .masks import (MaskBase, InvertedMask, CompositeMask,
                    BooleanArrayMask, LazyMask, LazyComparisonMask,
                    FunctionMask)
from .lower_dimensional_structures import (OneDSpectrum, Projection, Slice)

# Import the following sub-packages to make sure the I/O functions are registered
from .io import casa_image
del casa_image
from .io import class_lmv
del class_lmv
from .io import fits
del fits

__all__ = ['SpectralCube', 'VaryingResolutionSpectralCube',
           'DaskSpectralCube', 'DaskVaryingResolutionSpectralCube',
            'StokesSpectralCube', 'CompositeMask', 'LazyComparisonMask',
            'LazyMask', 'BooleanArrayMask', 'FunctionMask',
            'OneDSpectrum', 'Projection', 'Slice'
            ]
