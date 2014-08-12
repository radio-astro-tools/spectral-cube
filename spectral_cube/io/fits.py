import warnings

from astropy.io import fits
from astropy.wcs import WCS
from astropy.extern import six
from astropy.utils import OrderedDict
from astropy.io.fits.hdu.hdulist import fitsopen as fits_open

import numpy as np
from .. import SpectralCube, StokesSpectralCube, LazyMask
from .. import cube_utils

def first(iterable):
    return next(iter(iterable))


# FITS registry code - once Astropy includes a proper extensible I/O base
# class, we can use that instead. The following code takes care of
# interpreting string input (filename), HDU, and HDUList.

def is_fits(input, **kwargs):
    """
    Determine whether input is in FITS format
    """
    if isinstance(input, six.string_types):
        if input.lower().endswith(('.fits', '.fits.gz',
                                      '.fit', '.fit.gz',
                                      '.fits.Z', '.fit.Z')):
            return True
    elif isinstance(input, (fits.HDUList, fits.PrimaryHDU, fits.ImageHDU)):
        return True
    else:
        return False


def read_data_fits(input, hdu=None, **kwargs):
    """
    Read an array and header from an FITS file.

    Parameters
    ----------
    input : str or compatible `astropy.io.fits` HDU object
        If a string, the filename to read the table from. The
        following `astropy.io.fits` HDU objects can be used as input:
        - :class:`~astropy.io.fits.hdu.table.PrimaryHDU`
        - :class:`~astropy.io.fits.hdu.table.ImageHDU`
        - :class:`~astropy.io.fits.hdu.hdulist.HDUList`
    hdu : int or str, optional
        The HDU to read the table from.
    """

    if isinstance(input, fits.HDUList):

        # Parse all array objects
        arrays = OrderedDict()
        for ihdu, hdu_item in enumerate(input):
            if isinstance(hdu_item, (fits.PrimaryHDU, fits.ImageHDU)):
                arrays[ihdu] = hdu_item

        if len(arrays) > 1:
            if hdu is None:
                hdu = first(arrays)
                warnings.warn("hdu= was not specified but multiple arrays"
                              " are present, reading in first available"
                              " array (hdu={0})".format(hdu))

            # hdu might not be an integer, so we first need to convert it
            # to the correct HDU index
            hdu = input.index_of(hdu)

            if hdu in arrays:
                array_hdu = arrays[hdu]
            else:
                raise ValueError("No array found in hdu={0}".format(hdu))

        elif len(arrays) == 1:
            array_hdu = arrays[first(arrays)]
        else:
            raise ValueError("No table found")

    elif isinstance(input, (fits.PrimaryHDU, fits.ImageHDU)):

        array_hdu = input

    else:

        hdulist = fits_open(input, **kwargs)

        try:
            return read_data_fits(hdulist, hdu=hdu)
        finally:
            hdulist.close()

    return array_hdu.data, array_hdu.header


def load_fits_cube(input, hdu=0):
    """
    Read in a cube from a FITS file using astropy.

    Parameters
    ----------
    input: str or HDU
        The FITS cube file name or HDU
    hdu: int
        The extension number containing the data to be read
    """

    data, header = read_data_fits(input, hdu=hdu)
    meta = {}

    if 'BUNIT' in header:
        meta['BUNIT'] = header['BUNIT']

    wcs = WCS(header)

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


def write_fits_cube(filename, cube, overwrite=False):
    """
    Write a FITS cube with a WCS to a filename
    """

    if isinstance(cube, SpectralCube):
        cube.hdu.writeto(filename, clobber=overwrite)
    else:
        raise NotImplementedError()
