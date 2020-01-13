# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ._astropy_init import __version__, test

from pkg_resources import get_distribution, DistributionNotFound

from .spectral_cube import (SpectralCube, VaryingResolutionSpectralCube)
from .stokes_spectral_cube import StokesSpectralCube
from .masks import (MaskBase, InvertedMask, CompositeMask,
                    BooleanArrayMask, LazyMask, LazyComparisonMask,
                    FunctionMask)
from .lower_dimensional_structures import (OneDSpectrum, Projection, Slice)

__all__ = ['SpectralCube', 'VaryingResolutionSpectralCube',
            'StokesSpectralCube', 'CompositeMask', 'LazyComparisonMask',
            'LazyMask', 'BooleanArrayMask', 'FunctionMask',
            'OneDSpectrum', 'Projection', 'Slice'
            ]
