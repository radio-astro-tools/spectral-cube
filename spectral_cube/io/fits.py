from __future__ import print_function, absolute_import, division

import six
import warnings

from astropy.io import fits
import astropy.wcs
from astropy import wcs
from astropy.wcs import WCS
from collections import OrderedDict
from astropy.io.fits.hdu.hdulist import fitsopen as fits_open

import numpy as np
import datetime
try:
    from .. import version
    SPECTRAL_CUBE_VERSION = version.version
except ImportError:
    # We might be running py.test on a clean checkout
    SPECTRAL_CUBE_VERSION = 'dev'

from .. import SpectralCube, StokesSpectralCube, LazyMask, VaryingResolutionSpectralCube
from ..spectral_cube import BaseSpectralCube
from .. import cube_utils
from ..utils import FITSWarning, FITSReadError


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


def read_data_fits(input, hdu=None, mode='denywrite', **kwargs):
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
    mode : str
        One of the FITS file reading modes; see `~astropy.io.fits.open`.
        ``denywrite`` is used by default since this prevents the system from
        checking that the entire cube will fit into swap, which can prevent the
        file from being opened at all.
    """

    beam_table = None

    if isinstance(input, fits.HDUList):

        # Parse all array objects
        arrays = OrderedDict()
        for ihdu, hdu_item in enumerate(input):
            if isinstance(hdu_item, (fits.PrimaryHDU, fits.ImageHDU)):
                arrays[ihdu] = hdu_item
            elif isinstance(hdu_item, fits.BinTableHDU):
                if 'BPA' in hdu_item.data.names:
                    beam_table = hdu_item.data

        if len(arrays) > 1:
            if hdu is None:
                hdu = first(arrays)
                warnings.warn("hdu= was not specified but multiple arrays"
                              " are present, reading in first available"
                              " array (hdu={0})".format(hdu),
                              FITSWarning
                             )

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
            raise ValueError("No arrays found")

    elif isinstance(input, (fits.PrimaryHDU, fits.ImageHDU)):

        array_hdu = input

    else:

        with fits_open(input, mode=mode, **kwargs) as hdulist:
            return read_data_fits(hdulist, hdu=hdu)

    return array_hdu.data, array_hdu.header, beam_table


def load_fits_cube(input, hdu=0, meta=None, **kwargs):
    """
    Read in a cube from a FITS file using astropy.

    Parameters
    ----------
    input: str or HDU
        The FITS cube file name or HDU
    hdu: int
        The extension number containing the data to be read
    meta: dict
        Metadata (can be inherited from other readers, for example)
    """

    data, header, beam_table = read_data_fits(input, hdu=hdu, **kwargs)

    if data is None:
        raise FITSReadError('No data found in HDU {0}. You can try using the hdu= '
                            'keyword argument to read data from another HDU.'.format(hdu))

    if meta is None:
        meta = {}

    if 'BUNIT' in header:
        meta['BUNIT'] = header['BUNIT']

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                category=astropy.wcs.FITSFixedWarning,
                                append=True)
        wcs = WCS(header)

    if wcs.wcs.naxis == 3:

        data, wcs = cube_utils._orient(data, wcs)

        mask = LazyMask(np.isfinite, data=data, wcs=wcs)
        assert data.shape == mask._data.shape
        if beam_table is None:
            cube = SpectralCube(data, wcs, mask, meta=meta, header=header)
        else:
            cube = VaryingResolutionSpectralCube(data, wcs, mask, meta=meta,
                                                 header=header,
                                                 beam_table=beam_table)

        if hasattr(cube._mask, '_data'):
            # check that the shape matches if there is a shape
            # it is possible that VaryingResolution cubes will have a composite
            # mask instead
            assert cube._data.shape == cube._mask._data.shape

    elif wcs.wcs.naxis == 4:

        data, wcs = cube_utils._split_stokes(data, wcs)

        stokes_data = {}
        for component in data:
            comp_data, comp_wcs = cube_utils._orient(data[component], wcs)
            comp_mask = LazyMask(np.isfinite, data=comp_data, wcs=comp_wcs)
            if beam_table is None:
                stokes_data[component] = SpectralCube(comp_data, wcs=comp_wcs,
                                                      mask=comp_mask, meta=meta,
                                                      header=header)
            else:
                VRSC = VaryingResolutionSpectralCube
                stokes_data[component] = VRSC(comp_data, wcs=comp_wcs,
                                              mask=comp_mask, meta=meta,
                                              header=header,
                                              beam_table=beam_table)

        cube = StokesSpectralCube(stokes_data)

    else:

        raise FITSReadError("Data should be 3- or 4-dimensional")

    return cube


def write_fits_cube(filename, cube, overwrite=False,
                    include_origin_notes=True):
    """
    Write a FITS cube with a WCS to a filename
    """

    if isinstance(cube, BaseSpectralCube):
        hdulist = cube.hdulist
        now = datetime.datetime.strftime(datetime.datetime.now(),
                                         "%Y/%m/%d-%H:%M:%S")
        hdulist[0].header.add_history("Written by spectral_cube v{version} on "
                                      "{date}".format(version=SPECTRAL_CUBE_VERSION,
                                                      date=now))
        try:
            hdulist.writeto(filename, overwrite=overwrite)
        except TypeError:
            hdulist.writeto(filename, clobber=overwrite)
    else:
        raise NotImplementedError()
