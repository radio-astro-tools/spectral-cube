from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.wcs import (WCSSUB_SPECTRAL, WCSSUB_LONGITUDE, WCSSUB_LATITUDE)
from . import wcs_utils
from astropy import log
from astropy.io.fits import BinTableHDU, Column
from astropy import units as u
import itertools

def _fix_spectral(wcs):
    """
    Attempt to fix a cube with an invalid spectral axis definition.  Only uses
    well-known exceptions, e.g. CTYPE = 'VELOCITY'.  For the rest, it will try
    to raise a helpful error.
    """

    axtypes = wcs.get_axis_types()

    types = [a['coordinate_type'] for a in axtypes]

    if wcs.naxis not in (3,4):
        raise TypeError("The WCS has {0} axes of types {1}".format(len(types),
                                                                   types))

    # sanitize noncompliant headers
    if 'spectral' not in types:
        log.warning("No spectral axis found; header may be non-compliant.")
        for ind,tp in enumerate(types):
            if tp not in ('celestial','stokes'):
                if wcs.wcs.ctype[ind] in wcs_utils.bad_spectypes_mapping:
                    wcs.wcs.ctype[ind] = wcs_utils.bad_spectypes_mapping[wcs.wcs.ctype[ind]]

    return wcs

def _split_stokes(array, wcs):
    """
    Given a 4-d data cube with 4-d WCS (spectral cube + stokes) return a
    dictionary of data and WCS objects for each Stokes component

    Parameters
    ----------
    array : `~numpy.ndarray`
        The input 3-d array with two position dimensions, one spectral
        dimension, and a Stokes dimension.
    wcs : `~astropy.wcs.WCS`
        The input 3-d WCS with two position dimensions, one spectral
        dimension, and a Stokes dimension.
    """

    if array.ndim not in (3,4):
        raise ValueError("Input array must be 3- or 4-dimensional for a"
                         " STOKES cube")

    if wcs.wcs.naxis != 4:
        raise ValueError("Input WCS must be 4-dimensional for a STOKES cube")

    wcs = _fix_spectral(wcs)

    # reverse from wcs -> numpy convention
    axtypes = wcs.get_axis_types()[::-1]

    types = [a['coordinate_type'] for a in axtypes]

    try:
        # Find stokes dimension
        stokes_index = types.index('stokes')
    except ValueError:
        # stokes not in list, but we are 4d
        if types.count('celestial') == 2 and types.count('spectral') == 1:
            if None in types:
                stokes_index = types.index(None)
                log.warning("FITS file has no STOKES axis, but it has a blank"
                         " axis type at index {0} that is assumed to be "
                         "stokes.".format(4-stokes_index))
            else:
                for ii,tp in enumerate(types):
                    if tp not in ('celestial', 'spectral'):
                        stokes_index = ii
                        stokes_type = tp

                log.warning("FITS file has no STOKES axis, but it has an axis"
                         " of type {1} at index {0} that is assumed to be "
                         "stokes.".format(4-stokes_index, stokes_type))
        else:
            raise IOError("There are 4 axes in the data cube but no STOKES "
                          "axis could be identified")

    # TODO: make the stokes names more general
    stokes_names = ["I", "Q", "U", "V"]

    stokes_arrays = {}

    wcs_slice = wcs_utils.drop_axis(wcs, wcs.naxis - 1 - stokes_index)

    if array.ndim == 4:
        for i_stokes in range(array.shape[stokes_index]):

            array_slice = [i_stokes if idim == stokes_index else slice(None)
                           for idim in range(array.ndim)]

            stokes_arrays[stokes_names[i_stokes]] = array[array_slice]
    else:
        # 3D array with STOKES as a 4th header parameter
        stokes_arrays['I'] = array

    return stokes_arrays, wcs_slice


def _orient(array, wcs):
    """
    Given a 3-d spectral cube and WCS, swap around the axes so that the
    spectral axis cube is the first in Numpy notation, and the last in WCS
    notation.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The input 3-d array with two position dimensions and one spectral
        dimension.
    wcs : `~astropy.wcs.WCS`
        The input 3-d WCS with two position dimensions and one spectral
        dimension.
    """

    if array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")

    if wcs.wcs.naxis != 3:
        raise ValueError("Input WCS must be 3-dimensional")

    wcs = wcs_utils.diagonal_wcs_to_cdelt(_fix_spectral(wcs))

    # reverse from wcs -> numpy convention
    axtypes = wcs.get_axis_types()[::-1]

    types = [a['coordinate_type'] for a in axtypes]

    nums = [None if a['coordinate_type'] != 'celestial' else a['number']
            for a in axtypes]

    if 'stokes' in types:
        raise ValueError("Input WCS should not contain stokes")

    t = [types.index('spectral'), nums.index(1), nums.index(0)]
    result_array = array.transpose(t)

    result_wcs = wcs.sub([WCSSUB_LONGITUDE, WCSSUB_LATITUDE, WCSSUB_SPECTRAL])

    return result_array, result_wcs


def slice_syntax(f):
    """
    This decorator wraps a function that accepts a tuple of slices.

    After wrapping, the function acts like a property that accepts
    bracket syntax (e.g., p[1:3, :, :])

    Parameters
    ----------
    f : function
    """

    def wrapper(self):
        result = SliceIndexer(f, self)
        result.__doc__ = f.__doc__
        return result

    wrapper.__doc__ = slice_doc.format(f.__doc__ or '',
                                       f.__name__)

    result = property(wrapper)
    return result

slice_doc = """
{0}

Notes
-----
Supports efficient Numpy slice notation,
like ``{1}[0:3, :, 2:4]``
"""


class SliceIndexer(object):

    def __init__(self, func, _other):
        self._func = func
        self._other = _other

    def __getitem__(self, view):
        return self._func(self._other, view)

    def __iter__(self):
        raise Exception("You need to specify a slice (e.g. ``[:]`` or "
                        "``[0,:,:]`` in order to access this property.")


# TODO: make this into a proper configuration item
# TODO: make threshold depend on memory?
MEMORY_THRESHOLD=1e8

def is_huge(cube):
    if cube.size < MEMORY_THRESHOLD:  # smallish
        return False
    else:
        return True


def iterator_strategy(cube, axis=None):
    """
    Guess the most efficient iteration strategy
    for iterating over a cube, given its size and layout

    Parameters
    ----------
    cube : SpectralCube instance
        The cube to iterate over
    axis : [0, 1, 2]
        For reduction methods, the axis that is
        being collapsed

    Returns
    -------
    strategy : ['cube' | 'ray' | 'slice']
        The recommended iteration strategy.
        *cube* recommends working with the entire array in memory
        *slice* recommends working with one slice at a time
        *ray*  recommends working with one ray at a time
    """
    # pretty simple for now
    if cube.size < 1e8:  # smallish
        return 'cube'
    return 'slice'


def beams_to_bintable(beams):
    """
    Convert a list of beams to a CASA-style BinTableHDU
    """

    c1 = Column(name='BMAJ', format='1E', array=[bm.major.to(u.arcsec).value for bm in beams], unit=u.arcsec.to_string('FITS'))
    c2 = Column(name='BMIN', format='1E', array=[bm.minor.to(u.arcsec).value for bm in beams], unit=u.arcsec.to_string('FITS'))
    c3 = Column(name='BPA', format='1E', array=[bm.pa.to(u.deg).value for bm in beams], unit=u.deg.to_string('FITS'))
    c4 = Column(name='CHAN', format='1J', array=[bm.meta['CHAN'] if 'CHAN' in bm.meta else 0 for bm in beams])
    c5 = Column(name='POL', format='1J', array=[bm.meta['POL'] if 'POL' in bm.meta else 0 for bm in beams])

    bmhdu = BinTableHDU.from_columns([c1, c2, c3, c4, c5])
    bmhdu.header['EXTNAME'] = 'BEAMS'
    bmhdu.header['EXTVER'] = 1
    bmhdu.header['XTENSION'] = 'BINTABLE'
    return bmhdu

def average_beams(beams, includemask=None):
    """
    Average the beam major, minor, and PA attributes.

    This is usually a dumb thing to do!
    """

    from radio_beam import Beam
    from astropy.stats import circmean

    if includemask is None:
        includemask = itertools.cycle([True])

    major = u.Quantity([bm.major for bm,incl in zip(beams,includemask) if incl], u.deg)
    minor = u.Quantity([bm.minor for bm,incl in zip(beams,includemask) if incl], u.deg)
    pa = u.Quantity([bm.pa for bm,incl in zip(beams,includemask) if incl], u.deg)
    new_beam = Beam(major=major.mean(), minor=minor.mean(),
                    pa=circmean(pa, weights=major/minor))

    return new_beam
