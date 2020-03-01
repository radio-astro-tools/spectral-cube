import tempfile
import numpy as np
from astropy.wcs import WCS

__all__ = ['wcs_casa2astropy']


def wcs_casa2astropy(ndim, coordsys):
    """
    Convert a casac.coordsys object into an astropy.wcs.WCS object
    """

    # Rather than try and parse the CASA coordsys ourselves, we delegate
    # to CASA by getting it to write out a FITS file and reading back in
    # using WCS

    try:
        from casatools import image
    except ImportError:
        try:
            from taskinit import iatool as image
        except ImportError:
            raise ImportError("Could not import CASA (casac) and therefore cannot convert csys to WCS")

    tmpimagefile = tempfile.mktemp() + '.image'
    tmpfitsfile = tempfile.mktemp() + '.fits'
    ia = image()
    ia.fromarray(outfile=tmpimagefile,
                 pixels=np.ones([1] * ndim),
                 csys=coordsys, log=False)
    ia.done()

    ia.open(tmpimagefile)
    ia.tofits(tmpfitsfile, stokeslast=False)
    ia.done()

    return WCS(tmpfitsfile)
