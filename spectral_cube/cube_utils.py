import numpy as np
from . import wcs_utils


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

        array_slice = (i_stokes if idim == stokes_index else slice(None) for idim in range(array.ndim))

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
    nums = [None if a['coordinate_type'] != 'celestial' else a['number']
            for a in axtypes]

    if 'stokes' in types:
        raise ValueError("Input WCS should not contain stokes")

    t = [types.index('spectral'), nums.index(1), nums.index(0)]

    result_array = array.transpose(t)

    # TODO: Swap around WCS
    result_wcs = wcs

    return result_array, result_wcs
