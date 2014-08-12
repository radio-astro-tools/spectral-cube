import numpy as np
from . import wcs_utils
import warnings


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

    if array.ndim != 4:
        raise ValueError("Input array must be 4-dimensional")

    if wcs.wcs.naxis != 4:
        raise ValueError("Input WCS must be 4-dimensional")

    # reverse from wcs -> numpy convention
    axtypes = wcs.get_axis_types()[::-1]

    types = [a['coordinate_type'] for a in axtypes]

    # Find stokes dimension
    stokes_index = types.index('stokes')

    # TODO: make the stokes names more general
    stokes_names = ["I", "Q", "U", "V"]

    stokes_arrays = {}

    wcs_slice = wcs_utils.drop_axis(wcs, array.ndim - 1 - stokes_index)

    for i_stokes in range(array.shape[stokes_index]):

        array_slice = [i_stokes if idim == stokes_index else slice(None) for idim in range(array.ndim)]

        stokes_arrays[stokes_names[i_stokes]] = array[array_slice]

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

    # reverse from wcs -> numpy convention
    axtypes = wcs.get_axis_types()[::-1]

    types = [a['coordinate_type'] for a in axtypes]

    # sanitize noncompliant headers
    if 'spectral' not in types:
        warnings.warn("No spectral axis found; header may be non-compliant.")
        types = [tp if tp == 'celestial' else 'spectral' for tp in types]
        spec = types.index('spectral')
        if wcs.wcs.ctype[spec] in wcs_utils.bad_spectypes_mapping:
            wcs.wcs.ctype[spec] = wcs_utils.bad_spectypes_mapping[wcs.wcs.ctype[spec]]

    nums = [None if a['coordinate_type'] != 'celestial' else a['number']
            for a in axtypes]

    if 'stokes' in types:
        raise ValueError("Input WCS should not contain stokes")

    t = [types.index('spectral'), nums.index(1), nums.index(0)]
    result_array = array.transpose(t)

    t = wcs.wcs.naxis - np.array(t[::-1]) - 1
    result_wcs = wcs_utils.reindex_wcs(wcs, t)

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
        raise Exception("You need to specify a slice (e.g. ``[:]`` or ``[0,:,:]`` in order to access this property.")


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
    if np.product(cube.shape) < 1e8:  # smallish
        return 'cube'
    return 'slice'
