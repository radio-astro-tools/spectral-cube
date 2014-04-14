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

    metadata = {'filename':filename, 'extension_number':extnum}

    #if dropdeg:
    #    dropaxes = [ii for ii,dim in enumerate(data.shape) if dim==1]
    #    ndeg = len(dropaxes)
    #    if data.ndim - ndeg < 3:
    #        raise ValueError("Data has fewer than 3 non-degenerate axes and is therefore not a cube.")
    #    if ndeg > 0:
    #        metadata['degenerate_axes'] = dropaxes

    #    # squeeze returns a view so this is OK
    #    data = data.squeeze()
    #    for d in dropaxes:
    #        wcs = wcs_manipulation.drop_axis(wcs, d)

    mask = SpectralCubeMask(wcs, np.logical_not(valid))
    cube = SpectralCube(data, wcs, mask, metadata=metadata)
    return cube
    
    
