import numpy as np
from astropy.wcs import WCS

def drop_axis(wcs, dropax):
    """
    Drop the ax on axis dropax
    """
    inds = range(wcs.wcs.naxis)
    inds.pop(dropax)
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def wcs_swapaxes(wcs, ax0, ax1):
    """
    Swap axes in a WCS
    """
    inds = range(wcs.wcs.naxis)
    inds[ax0],inds[ax1] = inds[ax1],inds[ax0]
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def reindex_wcs(wcs, inds):
    outwcs = WCS(naxis=len(inds))

    cdelt = wcs.wcs.get_cdelt()
    pc = wcs.wcs.get_pc()

    outwcs.wcs.crpix = wcs.wcs.crpix[inds]
    outwcs.wcs.cdelt = cdelt[inds]
    outwcs.wcs.crval = wcs.wcs.crval[inds]
    outwcs.wcs.cunit = [wcs.wcs.cunit[i] for i in inds]
    outwcs.wcs.ctype = [wcs.wcs.ctype[i] for i in inds]
    outwcs.wcs.pc = pc[inds[:,None],inds[None,:]]
    outwcs.wcs.velosys = wcs.wcs.velosys

    return outwcs


def test_wcs_manipulation():
    pass
