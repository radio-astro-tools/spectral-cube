from __future__ import print_function, absolute_import, division

import six
import dask
import warnings

from astropy.io import fits
from astropy.io import registry as io_registry
import astropy.wcs
from astropy.wcs import WCS
from collections import OrderedDict
from astropy.io.fits.hdu.hdulist import fitsopen as fits_open
from astropy.io.fits.connect import FITS_SIGNATURE
from astropy import units as u

import numpy as np
import datetime
try:
    from .. import version
    SPECTRAL_CUBE_VERSION = version.version
except ImportError:
    # We might be running py.test on a clean checkout
    SPECTRAL_CUBE_VERSION = 'dev'

from .. import SpectralCube, StokesSpectralCube, LazyMask, VaryingResolutionSpectralCube
from ..dask_spectral_cube import DaskSpectralCube, DaskVaryingResolutionSpectralCube
from ..lower_dimensional_structures import LowerDimensionalObject
from ..spectral_cube import BaseSpectralCube
from .. import cube_utils
from ..utils import FITSWarning, FITSReadError, StokesWarning, BeamWarning


def first(iterable):
    return next(iter(iterable))


def is_fits(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether `origin` is a FITS file.

    Parameters
    ----------
    origin : str or readable file-like object
        Path or file object containing a potential FITS file.

    Returns
    -------
    is_fits : bool
        Returns `True` if the given file is a FITS file.
    """
    if fileobj is not None:
        pos = fileobj.tell()
        sig = fileobj.read(30)
        fileobj.seek(pos)
        return sig == FITS_SIGNATURE
    elif filepath is not None:
        if filepath.lower().endswith(('.fits', '.fits.gz', '.fit', '.fit.gz',
                                      '.fts', '.fts.gz')):
            return True
    elif isinstance(args[0], (fits.HDUList, fits.ImageHDU, fits.PrimaryHDU)):
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

    beam_units = (u.arcsec, u.arcsec)

    if isinstance(input, fits.HDUList):

        # Parse all array objects
        arrays = OrderedDict()
        for ihdu, hdu_item in enumerate(input):
            if isinstance(hdu_item, (fits.PrimaryHDU, fits.ImageHDU)):
                arrays[ihdu] = hdu_item
            elif isinstance(hdu_item, fits.BinTableHDU):
                # Check for CASA-standard beam table
                if hdu_item.header.get('EXTNAME') == 'BEAMS':
                    beam_table = hdu_item.data

                    # Check that the table has the expected form for beam units:
                    # 1: BMAJ 2: BMIN 3: BPA
                    for i in range(1, 4):
                        key = f"TUNIT{i}"
                        if key not in hdu_item.header:
                            warnings.warn(BeamWarning(f"Missing beam units keyword {key}"
                                                      " in the header."))

                    # Read the bmaj/bmin units from the header
                    # (we still assume BPA is degrees because we've never seen an exceptional case)
                    # this will crash if there is no appropriate header info
                    maj_kw = [kw for kw, val in hdu_item.header.items() if val == 'BMAJ'][0]
                    min_kw = [kw for kw, val in hdu_item.header.items() if val == 'BMIN'][0]
                    try:
                        maj_unit = hdu_item.header[maj_kw.replace('TTYPE', 'TUNIT')]
                        min_unit = hdu_item.header[min_kw.replace('TTYPE', 'TUNIT')]
                    except KeyError:
                        # the default units, if unspecified (as from CASA <= 4.7.2, we think), are arcseconds
                        maj_unit = u.arcsec
                        min_unit = u.arcsec

                    # AIPS uses non-FITS-standard unit names; this catches the
                    # only case we've seen so far
                    if maj_unit == 'DEGREES':
                        maj_unit = 'degree'
                    if min_unit == 'DEGREES':
                        min_unit = 'degree'

                    maj_unit = u.Unit(maj_unit)
                    min_unit = u.Unit(min_unit)

                    beam_units = (maj_unit, min_unit)

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

        if hasattr(input, 'read'):
            mode = None

        with fits_open(input, mode=mode, **kwargs) as hdulist:
            return read_data_fits(hdulist, hdu=hdu)

    return array_hdu.data, array_hdu.header, beam_table, beam_units


def load_fits_cube(input, hdu=0, meta=None, target_cls=None, use_dask=False, **kwargs):
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

    if use_dask:
        SC = DaskSpectralCube
        VRSC = DaskVaryingResolutionSpectralCube
    else:
        SC = SpectralCube
        VRSC = VaryingResolutionSpectralCube

    data, header, beam_table, beam_units = read_data_fits(input, hdu=hdu, **kwargs)

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
            cube = SC(data, wcs, mask, meta=meta, header=header)
        else:
            cube = VRSC(data, wcs, mask, meta=meta, header=header,
                        beam_table=beam_table, major_unit=beam_units[0],
                        minor_unit=beam_units[1])

        if hasattr(cube._mask, '_data'):
            # check that the shape matches if there is a shape
            # it is possible that VaryingResolution cubes will have a composite
            # mask instead
            assert cube._data.shape == cube._mask._data.shape

    elif wcs.wcs.naxis == 4:

        if beam_table is None:
            data, wcs = cube_utils._split_stokes(data, wcs)
        else:
            data, wcs, beam_tables = cube_utils._split_stokes(data, wcs, beam_table=beam_table)

        stokes_data = {}
        for component in data:
            comp_data, comp_wcs = cube_utils._orient(data[component], wcs)
            comp_mask = LazyMask(np.isfinite, data=comp_data, wcs=comp_wcs)
            if beam_table is None:
                stokes_data[component] = SC(comp_data, wcs=comp_wcs,
                                            mask=comp_mask, meta=meta,
                                            header=header)
            else:
                stokes_data[component] = VRSC(comp_data, wcs=comp_wcs,
                                              mask=comp_mask, meta=meta,
                                              header=header,
                                              beam_table=beam_tables[component],
                                              major_unit=beam_units[0],
                                              minor_unit=beam_units[1]
                                             )

        cube = StokesSpectralCube(stokes_data)

    else:

        raise FITSReadError("Data should be 3- or 4-dimensional")

    from .core import normalize_cube_stokes
    return normalize_cube_stokes(cube, target_cls=target_cls)


def write_fits_cube(cube, filename, overwrite=False,
                    include_origin_notes=True):
    """
    Write a FITS cube with a WCS to a filename
    """

    if isinstance(cube, BaseSpectralCube):
        hdulist = cube.hdulist
        if include_origin_notes:
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


def write_fits_ldo(data, filename, overwrite=False):
    # Spectra may have HDUList objects instead of HDUs because they
    # have a beam table attached, so we want to try that first
    # (a more elegant way to write this might be to do "self._hdu_general.write"
    # and create a property `self._hdu_general` that selects the right one...)
    if hasattr(data, 'hdulist'):
        try:
            data.hdulist.writeto(filename, overwrite=overwrite)
        except TypeError:
            data.hdulist.writeto(filename, clobber=overwrite)
    elif hasattr(data, 'hdu'):
        try:
            data.hdu.writeto(filename, overwrite=overwrite)
        except TypeError:
            data.hdu.writeto(filename, clobber=overwrite)


io_registry.register_reader('fits', BaseSpectralCube, load_fits_cube)
io_registry.register_writer('fits', BaseSpectralCube, write_fits_cube)
io_registry.register_identifier('fits', BaseSpectralCube, is_fits)

io_registry.register_reader('fits', StokesSpectralCube, load_fits_cube)
io_registry.register_writer('fits', StokesSpectralCube, write_fits_cube)
io_registry.register_identifier('fits', StokesSpectralCube, is_fits)

io_registry.register_writer('fits', LowerDimensionalObject, write_fits_ldo)
io_registry.register_identifier('fits', LowerDimensionalObject, is_fits)
