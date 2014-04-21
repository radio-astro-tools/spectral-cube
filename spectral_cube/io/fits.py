from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from .. import SpectralCube, StokesSpectralCube, LazyMask
from .. import cube_utils


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

    meta = {'filename': filename,
            'extension_number': extnum}

    return load_fits_hdu(hdulist[extnum], meta=meta)


def load_fits_hdu(hdu, meta={}, **kwargs):
    # read the data - assume first extension
    data = hdu.data

    # note the header and WCS information
    hdr = hdu.header
    wcs = WCS(hdr)

    if 'BUNIT' in hdr:
        meta['bunit'] = hdr['BUNIT']

    if wcs.wcs.naxis == 3:

        data, wcs = cube_utils._orient(data, wcs)

        mask = LazyMask(np.isfinite, data=data, wcs=wcs)
        cube = SpectralCube(data, wcs, mask, meta=meta)

    elif wcs.wcs.naxis == 4:

        data, wcs = cube_utils._split_stokes(data, wcs)

        mask = {}
        for component in data:
            data[component], wcs_slice = cube_utils._orient(data[component], wcs)
            mask[component] = LazyMask(np.isfinite, data=data[component], wcs=wcs_slice)

        cube = StokesSpectralCube(data, wcs_slice, mask, meta=meta)

    else:

        raise Exception("Data should be 3- or 4-dimensional")

    return cube


def write_fits_cube(filename, cube, include_stokes=False, clobber=False):
    """
    Write a FITS cube with a WCS to a filename

    TODO: document further!!
    """

    # TODO: add stokes axis on writing for CASA compatibility

    if isinstance(cube, SpectralCube) or (isinstance(cube, StokesSpectralCube) and not include_stokes):
        header = cube._wcs.to_header()
        outhdu = fits.PrimaryHDU(data=cube._data, header=header)
        outhdu.writeto(filename, clobber=clobber)
    else:
        raise NotImplementedError()
