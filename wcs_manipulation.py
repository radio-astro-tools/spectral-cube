import numpy as np
from astropy.wcs import WCS

def drop_axis(wcs, dropax):
    """
    Drop the ax on axis dropax
    """
    outwcs = WCS(naxis=2)

    inds = range(wcs.wcs.naxis)
    inds.pop(dropax)
    inds = np.array(inds)


    cdelt = wcs.wcs.get_cdelt()
    pc = wcs.wcs.get_pc()

    outwcs.wcs.crpix = wcs.wcs.crpix[inds]
    outwcs.wcs.cdelt = cdelt[inds]
    outwcs.wcs.crval = wcs.wcs.crval[inds]
    outwcs.wcs.cunit = [wcs.wcs.cunit[i] for i in inds]
    outwcs.wcs.ctype = [wcs.wcs.ctype[i] for i in inds]
    outwcs.wcs.pc = pc[inds[:,None],inds[None,:]]

    return outwcs
