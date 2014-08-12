from __future__ import print_function
import numpy as np
from astropy.wcs import WCS
import warnings
from astropy import units as u
from astropy import log

wcs_parameters_to_preserve = ['cel_offset', 'dateavg', 'dateobs', 'equinox',
                              'latpole', 'lonpole', 'mjdavg', 'mjdobs', 'name',
                              'obsgeo', 'phi0', 'radesys', 'restfrq',
                              'restwav', 'specsys', 'ssysobs', 'ssyssrc',
                              'theta0', 'velangl', 'velosys', 'zsource']
# not writable:
# 'lat', 'lng', 'lattyp', 'lngtyp',

bad_spectypes_mapping = {'VELOCITY':'VELO',
                         'WAVELENG':'WAVE',
                         }

def drop_axis(wcs, dropax):
    """
    Drop the ax on axis dropax

    Remove an axis from the WCS
    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS with naxis to be chopped to naxis-1
    dropax: int
        The index of the WCS to drop, counting from 0 (i.e., python convention,
        not FITS convention)
    """
    inds = list(range(wcs.wcs.naxis))
    inds.pop(dropax)
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def add_stokes_axis_to_wcs(wcs, add_before_ind):
    """
    Add a new Stokes axis that is uncorrelated with any other axes

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to add to
    add_before_ind: int
        Index of the WCS to insert the new Stokes axis in front of.
        To add at the end, do add_before_ind = wcs.wcs.naxis
    """

    naxin = wcs.wcs.naxis
    naxout = naxin + 1

    inds = list(range(naxout))
    inds.pop(add_before_ind)
    inds = np.array(inds)

    outwcs = WCS(naxis=naxout)
    for par in wcs_parameters_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs, par))

    pc = np.zeros([naxout, naxout])
    pc[inds[:, np.newaxis], inds[np.newaxis, :]] = wcs.wcs.get_pc()
    pc[add_before_ind, add_before_ind] = 1

    def append_to_end(val, lst):
        """ insert a value at index into a list """
        return list(lst) + [val]

    outwcs.wcs.crpix = append_to_end(1, wcs.wcs.crpix)
    outwcs.wcs.cdelt = append_to_end(1, wcs.wcs.get_cdelt())
    outwcs.wcs.crval = append_to_end(1, wcs.wcs.crval)
    outwcs.wcs.cunit = append_to_end("", wcs.wcs.cunit)
    outwcs.wcs.ctype = append_to_end("STOKES", wcs.wcs.ctype)
    outwcs.wcs.cname = append_to_end("STOKES", wcs.wcs.cname)
    outwcs.wcs.pc = pc

    return outwcs


def wcs_swapaxes(wcs, ax0, ax1):
    """
    Swap axes in a WCS

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to have its axes swapped
    ax0: int
    ax1: int
        The indices of the WCS to be swapped, counting from 0 (i.e., python
        convention, not FITS convention)
    """
    inds = list(range(wcs.wcs.naxis))
    inds[ax0], inds[ax1] = inds[ax1], inds[ax0]
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def reindex_wcs(wcs, inds):
    """
    Re-index a WCS given indices.  The number of axes may be reduced.

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to be manipulated
    inds: np.array(dtype='int')
        The indices of the array to keep in the output.
        e.g. swapaxes: [0,2,1,3]
        dropaxes: [0,1,3]
    """

    if not isinstance(inds, np.ndarray):
        raise TypeError("Indices must be an ndarray")

    if inds.dtype.kind != 'i':
        raise TypeError('Indices must be integers')

    outwcs = WCS(naxis=len(inds))
    for par in wcs_parameters_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs, par))

    cdelt = wcs.wcs.get_cdelt()
    pc = wcs.wcs.get_pc()

    outwcs.wcs.crpix = wcs.wcs.crpix[inds]
    outwcs.wcs.cdelt = cdelt[inds]
    outwcs.wcs.crval = wcs.wcs.crval[inds]
    outwcs.wcs.cunit = [wcs.wcs.cunit[i] for i in inds]
    outwcs.wcs.ctype = [wcs.wcs.ctype[i] for i in inds]
    outwcs.wcs.cname = [wcs.wcs.cname[i] for i in inds]
    outwcs.wcs.pc = pc[inds[:, None], inds[None, :]]

    return outwcs


def axis_names(wcs):
    """
    Extract world names for each coordinate axis

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to extract names from

    Returns
    -------
    A tuple of names along each axis
    """
    names = list(wcs.wcs.cname)
    types = wcs.wcs.ctype
    for i in range(len(names)):
        if len(names[i]) > 0:
            continue
        names[i] = types[i].split('-')[0]
    return names


def slice_wcs(wcs, view):
    """
    Slice a WCS instance using a Numpy slice. The order of the slice should
    be reversed (as for the data) compared to the natural WCS order.

    Parameters
    ----------
    view : tuple
        A tuple containing the same number of slices as the WCS system

    Returns
    -------
    A new `~astropy.wcs.WCS` instance
    """
    if len(view) != wcs.wcs.naxis:
        raise ValueError("Must have same number of slices as number of WCS axes")

    wcs_new = wcs.deepcopy()
    for i, iview in enumerate(view):
        if iview.start is not None:
            if iview.step not in (None, 1):
                raise NotImplementedError("Cannot yet slice WCS with strides different from None or 1")
            wcs_index = wcs.wcs.naxis - 1 - i
            wcs_new.wcs.crpix[wcs_index] -= iview.start
    return wcs_new

def check_equality(wcs1, wcs2, warn_missing=False, ignore_keywords=['MJD-OBS',
                                                                    'VELOSYS']):
    """
    Check if two WCSs are equal

    Parameters
    ----------
    wcs1, wcs2: `astropy.wcs.WCS`
        The WCSs
    warn_missing: bool
        Issue warnings if one header is missing a keyword that the other has?
    ignore_keywords: list of str
        Keywords that are stored as part of the WCS but do not define part of
        the coordinate system and therefore can be safely ignored.
    """

    # naive version:
    # return str(wcs1.to_header()) != str(wcs2.to_header())

    h1 = wcs1.to_header()
    h2 = wcs2.to_header()

    # Default to headers equal; everything below changes to false if there are
    # any inequalities
    OK = True
    # to keep track of keywords in both
    matched = []

    for c1 in h1.cards:
        key = c1[0]
        if key in h2:
            matched.append(key)
            c2 = h2.cards[key]
            # special check for units: "m/s" = "m s-1"
            if 'UNIT' in key:
                u1 = u.Unit(c1[1])
                u2 = u.Unit(c2[1])
                if u1 != u2:
                    if key in ignore_keywords:
                        log.debug("IGNORED Header 1, {0}: {1} != {2}".format(key,u1,u2))
                    else:
                        OK = False
                        log.debug("Header 1, {0}: {1} != {2}".format(key,u1,u2))
            elif isinstance(c1[1], (float, np.float)):
                try:
                    np.testing.assert_almost_equal(c1[1], c2[1])
                except AssertionError:
                    if key in ('RESTFRQ','RESTWAV'):
                        warnings.warn("{0} is not equal in WCS; ignoring ".format(key)+
                                      "under the assumption that you want to"
                                      " compare velocity cubes.")
                        continue
                    if key in ignore_keywords:
                        log.debug("IGNORED Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                    else:
                        log.debug("Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                        OK = False
            elif c1[1] != c2[1]:
                if key in ignore_keywords:
                    log.debug("IGNORED Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                else:
                    log.debug("Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                    OK = False
        else:
            if warn_missing:
                warnings.warn("WCS2 is missing card {0}".format(key))
            elif key not in ignore_keywords:
                OK = False

    # Check that there aren't any cards in header 2 that were missing from
    # header 1
    for c2 in h2.cards:
        key = c2[0]
        if key not in matched:
            if warn_missing:
                warnings.warn("WCS1 is missing card {0}".format(key))
            else:
                OK = False

    return OK
