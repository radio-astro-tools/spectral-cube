# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import __version__, __githash__, test
# ----------------------------------------------------------------------------
if not _ASTROPY_SETUP_:
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
