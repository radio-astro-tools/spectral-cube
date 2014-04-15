import warnings
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from spectral_cube import SpectralCube,SpectralCubeMask
from .. import wcs_manipulation


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

def write_fits_cube(filename, data, wcs, includestokes=False, clobber=False):
    """
    Write a FITS cube with a WCS to a filename

    TODO: document further!!
    """
    axtypes = wcs.get_axis_types()
    stokesax = np.array([a['coordinate_type'] == 'stokes' for a in axtypes])
    if not includestokes:
        if data.shape[0] != 1:
            raise ValueError("Cannot drop stokes unless it's degenerate")
        data = data[0,:,:,:]
        if np.any(stokesax):
            drop = np.argmax(stokesax)
            wcs = wcs_manipulation.drop_axis(wcs, drop)
    else:
        if not np.any(stokesax):
            wcs = wcs_manipulation.add_stokes_axis_to_wcs(wcs, 0)

    header = wcs.to_header()
    outhdu = fits.PrimaryHDU(data=data, header=header)
    outhdu.writeto(filename, clobber=clobber)
