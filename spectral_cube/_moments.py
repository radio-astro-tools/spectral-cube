from __future__ import print_function, absolute_import, division

import numpy as np

from .cube_utils import iterator_strategy
from .np_compat import allbadtonan

"""
Functions to compute moment maps in a variety of ways
"""


def _moment_shp(cube, axis):
    """
    Return the shape of the moment map

    Parameters
    -----------
    cube : SpectralCube
       The cube to collapse
    axis : int
       The axis to collapse along (numpy convention)

    Returns
    -------
    ny, nx
    """
    return cube.shape[:axis] + cube.shape[axis + 1:]


def _slice0(cube, axis):
    """
    0th moment along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int

    Returns
    -------
    moment0 : array
    """
    shp = _moment_shp(cube, axis)
    result = np.zeros(shp)

    view = [slice(None)] * 3

    valid = np.zeros(shp, dtype=bool)
    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = cube._get_filled_data(fill=np.nan, view=tuple(view))
        valid |= np.isfinite(plane)
        result += np.nan_to_num(plane) * cube._pix_size_slice(axis)
    result[~valid] = np.nan
    return result


def _slice1(cube, axis):
    """
    1st moment along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int

    Returns
    -------
    moment1 : array
    """
    shp = _moment_shp(cube, axis)
    result = np.zeros(shp)

    view = [slice(None)] * 3
    pix_size = cube._pix_size_slice(axis)
    pix_cen = cube._pix_cen()[axis]
    weights = np.zeros(shp)

    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = cube._get_filled_data(fill=0, view=tuple(view))
        result += (plane *
                   pix_cen[tuple(view)] *
                   pix_size)
        weights += plane * pix_size
    return result / weights


def moment_slicewise(cube, order, axis):
    """
    Compute moments by accumulating the result 1 slice at a time
    """
    if order == 0:
        return _slice0(cube, axis)
    if order == 1:
        return _slice1(cube, axis)

    shp = _moment_shp(cube, axis)
    result = np.zeros(shp)

    view = [slice(None)] * 3
    pix_size = cube._pix_size_slice(axis)
    pix_cen = cube._pix_cen()[axis]
    weights = np.zeros(shp)

    # would be nice to get mom1 and momn in single pass over data
    # possible for mom2, not sure about general case
    mom1 = _slice1(cube, axis)

    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = cube._get_filled_data(fill=0, view=tuple(view))
        result += (plane *
                   (pix_cen[tuple(view)] - mom1) ** order *
                   pix_size)
        weights += plane * pix_size

    return (result / weights)


def moment_raywise(cube, order, axis):
    """
    Compute moments by accumulating the answer one ray at a time
    """
    shp = _moment_shp(cube, axis)
    out = np.zeros(shp) * np.nan

    pix_cen = cube._pix_cen()[axis]
    pix_size = cube._pix_size_slice(axis)

    for x, y, slc in cube._iter_rays(axis):
        # the intensity, i.e. the weights
        include = cube._mask.include(data=cube._data, wcs=cube._wcs, view=slc,
                                     wcs_tolerance=cube._wcs_tolerance)
        if not include.any():
            continue

        data = cube.flattened(slc).value * pix_size

        if order == 0:
            out[x, y] = data.sum()
            continue

        order1 = (data * pix_cen[slc][include]).sum() / data.sum()
        if order == 1:
            out[x, y] = order1
            continue

        ordern = (data * (pix_cen[slc][include] - order1) ** order).sum()
        ordern /= data.sum()

        out[x, y] = ordern
    return out

def moment_cubewise(cube, order, axis):
    """
    Compute the moments by working with the entire data at once
    """

    pix_cen = cube._pix_cen()[axis]
    data = cube._get_filled_data() * cube._pix_size_slice(axis)

    if order == 0:
        return allbadtonan(np.nansum)(data, axis=axis)

    if order == 1:
        return (np.nansum(data * pix_cen, axis=axis) /
                np.nansum(data, axis=axis))
    else:
        mom1 = moment_cubewise(cube, 1, axis)

        # insert an axis so it broadcasts properly
        shp = list(_moment_shp(cube, axis))
        shp.insert(axis, 1)
        mom1 = mom1.reshape(shp)

        return (np.nansum(data * (pix_cen - mom1) ** order, axis=axis) /
                np.nansum(data, axis=axis))


def moment_auto(cube, order, axis):
    """
    Build a moment map, choosing a strategy to balance speed and memory.
    """
    strategy = dict(cube=moment_cubewise, ray=moment_raywise,
                    slice=moment_slicewise)
    return strategy[iterator_strategy(cube, axis)](cube, order, axis)
