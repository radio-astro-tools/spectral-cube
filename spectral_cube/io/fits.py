import warnings
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from spectral_cube import SpectralCube,SpectralCubeMask


def load_fits_cube(filename, extnum=0, **kwargs):
    """
    Read in a cube from a FITS file using astropy.

    Parameters
    ----------
    filename: str
        The FITS cube file name
    extnum: int
        The extension number containing the data to be read
    kwargs: dict
        passed to fits.open
    """

    # open the file
    hdulist = fits.open(filename, **kwargs)

    # read the data - assume first extension
    data = hdulist[extnum].data

    # note where data is valid
    valid = np.isfinite(data)

    # note the header and WCS information
    hdr = hdulist[extnum].header
    wcs = WCS(hdr)

    meta = {'filename': filename,
            'extension_number': extnum}

    mask = SpectralCubeMask(np.logical_not(valid), wcs)
    cube = SpectralCube(data, wcs, mask, meta=meta)

    return cube
