import contextlib
import warnings
import tempfile
import os
import time
from copy import deepcopy

import builtins

import dask.array as da
from dask.distributed import Client
import numpy as np
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.wcs import (WCSSUB_SPECTRAL, WCSSUB_LONGITUDE, WCSSUB_LATITUDE)
from astropy.wcs import WCS
from . import wcs_utils
from .utils import FITSWarning, AstropyUserWarning, WCSCelestialError
from astropy import log
from astropy.io import fits
from astropy.wcs.utils import is_proj_plane_distorted
from astropy.io.fits import BinTableHDU, Column
from astropy import units as u
import itertools
import re
from radio_beam import Beam
from radio_beam.utils import BeamError
from multiprocessing import Process, Pool

from functools import partial

def _fix_spectral(wcs):
    """
    Attempt to fix a cube with an invalid spectral axis definition.  Only uses
    well-known exceptions, e.g. CTYPE = 'VELOCITY'.  For the rest, it will try
    to raise a helpful error.
    """

    axtypes = wcs.get_axis_types()

    types = [a['coordinate_type'] for a in axtypes]

    if wcs.naxis not in (3, 4):
        raise TypeError("The WCS has {0} axes of types {1}".format(len(types),
                                                                   types))

    # sanitize noncompliant headers
    if 'spectral' not in types:
        log.warning("No spectral axis found; header may be non-compliant.")
        for ind,tp in enumerate(types):
            if tp not in ('celestial','stokes'):
                if wcs.wcs.ctype[ind] in wcs_utils.bad_spectypes_mapping:
                    wcs.wcs.ctype[ind] = wcs_utils.bad_spectypes_mapping[wcs.wcs.ctype[ind]]

    return wcs

def _split_stokes(array, wcs, beam_table=None):
    """
    Given a 4-d data cube with 4-d WCS (spectral cube + stokes) return a
    dictionary of data and WCS objects for each Stokes component

    Parameters
    ----------
    array : `~numpy.ndarray`
        The input 3-d array with two position dimensions, one spectral
        dimension, and a Stokes dimension.
    wcs : `~astropy.wcs.WCS`
        The input 3-d WCS with two position dimensions, one spectral
        dimension, and a Stokes dimension.
    beam_table : `~astropy.io.fits.hdu.table.BinTableHDU`
        When multiple beams are present, a FITS table with the beam information
        can be given to be split into the polarization components, consistent with
        `array`.
    """

    if array.ndim not in (3,4):
        raise ValueError("Input array must be 3- or 4-dimensional for a"
                         " STOKES cube")

    if wcs.wcs.naxis != 4:
        raise ValueError("Input WCS must be 4-dimensional for a STOKES cube")

    wcs = _fix_spectral(wcs)

    # reverse from wcs -> numpy convention
    axtypes = wcs.get_axis_types()[::-1]

    types = [a['coordinate_type'] for a in axtypes]

    try:
        # Find stokes dimension
        stokes_index = types.index('stokes')
    except ValueError:
        # stokes not in list, but we are 4d
        if types.count('celestial') == 2 and types.count('spectral') == 1:
            if None in types:
                stokes_index = types.index(None)
                log.warning("FITS file has no STOKES axis, but it has a blank"
                            " axis type at index {0} that is assumed to be "
                            "stokes.".format(4-stokes_index))
            else:
                for ii,tp in enumerate(types):
                    if tp not in ('celestial', 'spectral'):
                        stokes_index = ii
                        stokes_type = tp

                log.warning("FITS file has no STOKES axis, but it has an axis"
                            " of type {1} at index {0} that is assumed to be "
                            "stokes.".format(4-stokes_index, stokes_type))
        else:
            raise IOError("There are 4 axes in the data cube but no STOKES "
                          "axis could be identified")

    # TODO: make the stokes names more general
    stokes_names = ["I", "Q", "U", "V"]

    stokes_arrays = {}

    if beam_table is not None:
        beam_tables = {}

    wcs_slice = wcs_utils.drop_axis(wcs, wcs.naxis - 1 - stokes_index)

    if array.ndim == 4:
        for i_stokes in range(array.shape[stokes_index]):

            array_slice = [i_stokes if idim == stokes_index else slice(None)
                           for idim in range(array.ndim)]

            stokes_arrays[stokes_names[i_stokes]] = array[tuple(array_slice)]

            if beam_table is not None:
                beam_pol_idx = beam_table['POL'] == i_stokes
                beam_tables[stokes_names[i_stokes]] = beam_table[beam_pol_idx]

    else:
        # 3D array with STOKES as a 4th header parameter
        stokes_arrays['I'] = array

        if beam_table is not None:
            beam_tables['I'] = beam_table

    if beam_table is not None:
        return stokes_arrays, wcs_slice, beam_tables
    else:
        return stokes_arrays, wcs_slice


def _orient(array, wcs):
    """
    Given a 3-d spectral cube and WCS, swap around the axes so that the
    spectral axis cube is the first in Numpy notation, and the last in WCS
    notation.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The input 3-d array with two position dimensions and one spectral
        dimension.
    wcs : `~astropy.wcs.WCS`
        The input 3-d WCS with two position dimensions and one spectral
        dimension.
    """

    if array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")

    if wcs.wcs.naxis != 3:
        raise ValueError("Input WCS must be 3-dimensional")

    wcs = wcs_utils.diagonal_wcs_to_cdelt(_fix_spectral(wcs))

    # reverse from wcs -> numpy convention
    axtypes = wcs.get_axis_types()[::-1]

    types = [a['coordinate_type'] for a in axtypes]

    n_celestial = types.count('celestial')

    if n_celestial == 0:
        raise ValueError('No celestial axes found in WCS')
    elif n_celestial != 2:
        raise ValueError('WCS should contain 2 celestial dimensions but '
                         'contains {0}'.format(n_celestial))

    n_spectral = types.count('spectral')

    if n_spectral == 0:
        raise ValueError('No spectral axes found in WCS')
    elif n_spectral != 1:
        raise ValueError('WCS should contain one spectral dimension but '
                         'contains {0}'.format(n_spectral))

    nums = [None if a['coordinate_type'] != 'celestial' else a['number']
            for a in axtypes]

    if 'stokes' in types:
        raise ValueError("Input WCS should not contain stokes")

    t = [types.index('spectral'), nums.index(1), nums.index(0)]
    if t == [0, 1, 2]:
        result_array = array
    else:
        result_array = array.transpose(t)

    result_wcs = wcs.sub([WCSSUB_LONGITUDE, WCSSUB_LATITUDE, WCSSUB_SPECTRAL])

    return result_array, result_wcs


def slice_syntax(f):
    """
    This decorator wraps a function that accepts a tuple of slices.

    After wrapping, the function acts like a property that accepts
    bracket syntax (e.g., p[1:3, :, :])

    Parameters
    ----------
    f : function
    """

    def wrapper(self):
        result = SliceIndexer(f, self)
        result.__doc__ = f.__doc__
        return result

    wrapper.__doc__ = slice_doc.format(f.__doc__ or '',
                                       f.__name__)

    result = property(wrapper)

    return result

slice_doc = """
{0}

Notes
-----
Supports efficient Numpy slice notation,
like ``{1}[0:3, :, 2:4]``
"""


class SliceIndexer(object):

    def __init__(self, func, _other):
        self._func = func
        self._other = _other

    def __getitem__(self, view):
        result = self._func(self._other, view)
        if isinstance(result, da.Array):
            result = result.compute()
        return result

    @property
    def size(self):
        return self._other.size

    @property
    def ndim(self):
        return self._other.ndim

    @property
    def shape(self):
        return self._other.shape

    def __iter__(self):
        raise Exception("You need to specify a slice (e.g. ``[:]`` or "
                        "``[0,:,:]`` in order to access this property.")


# TODO: make this into a proper configuration item
# TODO: make threshold depend on memory?
MEMORY_THRESHOLD=1e8

def is_huge(cube):
    if cube.size < MEMORY_THRESHOLD:  # smallish
        return False
    else:
        return True


def iterator_strategy(cube, axis=None):
    """
    Guess the most efficient iteration strategy
    for iterating over a cube, given its size and layout

    Parameters
    ----------
    cube : SpectralCube instance
        The cube to iterate over
    axis : [0, 1, 2]
        For reduction methods, the axis that is
        being collapsed

    Returns
    -------
    strategy : ['cube' | 'ray' | 'slice']
        The recommended iteration strategy.
        *cube* recommends working with the entire array in memory
        *slice* recommends working with one slice at a time
        *ray*  recommends working with one ray at a time
    """
    # pretty simple for now
    if cube.size < 1e8:  # smallish
        return 'cube'
    return 'slice'


def try_load_beam(header):
    '''
    Try loading a beam from a FITS header.
    '''

    try:
        beam = Beam.from_fits_header(header)
        return beam
    except Exception as ex:
        # We don't emit a warning if no beam was found since it's ok for
        # cubes to not have beams
        # if 'No BMAJ' not in str(ex):
        #     warnings.warn("Could not parse beam information from header."
        #                   "  Exception was: {0}".format(ex.__repr__()),
        #                   FITSWarning
        #                  )

        # Avoid warning since cubes don't have a beam
        # Warning now provided when `SpectralCube.beam` is None
        beam = None

    return beam

def try_load_beams(data):
    '''
    Try loading a beam table from a FITS HDU list.
    '''
    try:
        from radio_beam import Beam
    except ImportError:
        warnings.warn("radio_beam is not installed. No beam "
                      "can be created.",
                      ImportError
                     )

    if isinstance(data, fits.BinTableHDU):
        if 'BPA' in data.data.names:
            beam_table = data.data
            return beam_table
        else:
            raise ValueError("No beam table found")
    elif isinstance(data, fits.HDUList):

        for ihdu, hdu_item in enumerate(data):
            if isinstance(hdu_item, (fits.PrimaryHDU, fits.ImageHDU)):
                beam = try_load_beams(hdu_item.header)
            elif isinstance(hdu_item, fits.BinTableHDU):
                if 'BPA' in hdu_item.data.names:
                    beam_table = hdu_item.data
                    return beam_table

        try:
            # if there was a beam in a header, but not a beam table
            return beam
        except NameError:
            # if the for loop has completed, we didn't find a beam table
            raise ValueError("No beam table found")
    elif isinstance(data, (fits.PrimaryHDU, fits.ImageHDU)):
        return try_load_beams(data.header)
    elif isinstance(data, fits.Header):
        try:
            beam = Beam.from_fits_header(data)
            return beam
        except Exception as ex:
            # warnings.warn("Could not parse beam information from header."
            #               "  Exception was: {0}".format(ex.__repr__()),
            #               FITSWarning
            #              )

            # Avoid warning since cubes don't have a beam
            # Warning now provided when `SpectralCube.beam` is None
            beam = None
    else:
        raise ValueError("How did you get here?  This is some sort of error.")


def beams_to_bintable(beams):
    """
    Convert a list of beams to a CASA-style BinTableHDU
    """

    c1 = Column(name='BMAJ', format='1E', array=[bm.major.to(u.arcsec).value for bm in beams], unit=u.arcsec.to_string('FITS'))
    c2 = Column(name='BMIN', format='1E', array=[bm.minor.to(u.arcsec).value for bm in beams], unit=u.arcsec.to_string('FITS'))
    c3 = Column(name='BPA', format='1E', array=[bm.pa.to(u.deg).value for bm in beams], unit=u.deg.to_string('FITS'))
    #c4 = Column(name='CHAN', format='1J', array=[bm.meta['CHAN'] if 'CHAN' in bm.meta else 0 for bm in beams])
    c4 = Column(name='CHAN', format='1J', array=np.arange(len(beams)))
    c5 = Column(name='POL', format='1J', array=[bm.meta['POL'] if 'POL' in bm.meta else 0 for bm in beams])

    bmhdu = BinTableHDU.from_columns([c1, c2, c3, c4, c5])
    bmhdu.header['EXTNAME'] = 'BEAMS'
    bmhdu.header['EXTVER'] = 1
    bmhdu.header['XTENSION'] = 'BINTABLE'
    bmhdu.header['NCHAN'] = len(beams)
    bmhdu.header['NPOL'] = len(set([bm.meta['POL'] for bm in beams if 'POL' in bm.meta]))
    return bmhdu


def beam_props(beams, includemask=None):
    '''
    Returns separate quantities for the major, minor, and PA of a list of
    beams.
    '''
    if includemask is None:
        includemask = itertools.cycle([True])

    major = u.Quantity([bm.major for bm, incl in zip(beams, includemask)
                        if incl], u.deg)
    minor = u.Quantity([bm.minor for bm, incl in zip(beams, includemask)
                        if incl], u.deg)
    pa = u.Quantity([bm.pa for bm, incl in zip(beams, includemask)
                     if incl], u.deg)

    return major, minor, pa


def largest_beam(beams, includemask=None):
    """
    Returns the largest beam (by area) in a list of beams.
    """

    from radio_beam import Beam

    major, minor, pa = beam_props(beams, includemask)
    largest_idx = (major * minor).argmax()
    new_beam = Beam(major=major[largest_idx], minor=minor[largest_idx],
                    pa=pa[largest_idx])

    return new_beam


def smallest_beam(beams, includemask=None):
    """
    Returns the smallest beam (by area) in a list of beams.
    """

    from radio_beam import Beam

    major, minor, pa = beam_props(beams, includemask)
    smallest_idx = (major * minor).argmin()
    new_beam = Beam(major=major[smallest_idx], minor=minor[smallest_idx],
                    pa=pa[smallest_idx])

    return new_beam


@contextlib.contextmanager
def _map_context(numcores):
    """
    Mapping context manager to allow parallel mapping or regular mapping
    depending on the number of cores specified.

    The builtin map is overloaded to handle python3 problems: python3 returns a
    generator, while ``multiprocessing.Pool.map`` actually runs the whole thing
    """
    if numcores is not None and numcores > 1:
        try:
            from joblib import Parallel, delayed
            from joblib.pool import has_shareable_memory
            map = lambda x,y: Parallel(n_jobs=numcores)(delayed(has_shareable_memory)(x))(y)
            parallel = True
        except ImportError:
            map = lambda x,y: list(builtins.map(x,y))
            warnings.warn("Could not import joblib.  "
                          "map will be non-parallel.",
                          ImportError
                         )
            parallel = False
    else:
        parallel = False
        map = lambda x,y: list(builtins.map(x,y))

    yield map


def convert_bunit(bunit):
    '''
    Convert a BUNIT string to a quantity

    Parameters
    ----------
    bunit : str
        String to convert to an `~astropy.units.Unit`

    Returns
    -------
    unit : `~astropy.unit.Unit`
        Corresponding unit.
    '''

    # special case: CASA (sometimes) makes non-FITS-compliant jy/beam headers
    bunit_lower = re.sub(r"\s", "", bunit.lower())
    if bunit_lower == 'jy/beam':
        unit = u.Jy / u.beam
    else:
        try:
            unit = u.Unit(bunit)
        except ValueError:
            warnings.warn("Could not parse unit {0}. "
                    "If you know the correct unit, try "
                    "u.add_enabled_units(u.def_unit(['{0}'], represents=u.<correct_unit>))".format(bunit),
                          AstropyUserWarning)
            unit = None

    return unit


def world_take_along_axis(cube, position_plane, axis):
    '''
    Convert a 2D plane of pixel positions to the equivalent WCS coordinates.
    For example, this will convert `argmax`
    along the spectral axis to the equivalent spectral value (e.g., velocity at
    peak intensity).

    Parameters
    ----------
    cube : SpectralCube
        A spectral cube.
    position_plane : 2D numpy.ndarray
        2D array of pixel positions along `axis`. For example, `position_plane` can
        be the output of `argmax` or `argmin` along an axis.
    axis : int
        The axis that `position_plane` is collapsed along.

    Returns
    -------
    out : astropy.units.Quantity
        2D array of WCS coordinates.
    '''

    if wcs_utils.is_pixel_axis_to_wcs_correlated(cube.wcs, axis):
        raise WCSCelestialError("world_take_along_axis requires the celestial axes"
                                " to be aligned along image axes.")

    # Get 1D slice along that axis.
    world_slice = [0, 0]
    world_slice.insert(axis, slice(None))

    world_coords = cube.world[tuple(world_slice)][axis]

    world_newaxis = [np.newaxis] * 2
    world_newaxis.insert(axis, slice(None))
    world_newaxis = tuple(world_newaxis)

    plane_newaxis = [slice(None), slice(None)]
    plane_newaxis.insert(axis, np.newaxis)
    plane_newaxis = tuple(plane_newaxis)

    out = np.take_along_axis(world_coords[world_newaxis],
                             position_plane[plane_newaxis], axis=axis)
    out = out.squeeze()

    return out


def _has_beam(obj):
    if hasattr(obj, '_beam'):
        return obj._beam is not None
    else:
        return False


def _has_beams(obj):
    if hasattr(obj, '_beams'):
        return obj._beams is not None
    else:
        return False


def bunit_converters(obj, unit, equivalencies=(), freq=None):
    '''
    Handler for all brightness unit conversions, including: K, Jy/beam, Jy/pix, Jy/sr.
    This also includes varying resolution spectral cubes, where the beam size varies along
    the frequency axis.

    Parameters
    ----------
    obj : {SpectralCube, LowerDimensionalObject}
        A spectral cube or any other lower dimensional object.
    unit : `~astropy.units.Unit`
        Unit to convert `obj` to.
    equivalencies : tuple, optional
        Initial list of equivalencies.
    freq : `~astropy.unit.Quantity`, optional
        Frequency to use for spectral conversions. If the spectral axis is available, the
        frequencies will already be defined.

    Outputs
    -------
    factor : `~numpy.ndarray`
        Array of factors for the unit conversion.

    '''

    # Add a simple check it the new unit is already equivalent, and so we don't need
    # any additional unit equivalencies
    if obj.unit.is_equivalent(unit):
        # return equivalencies
        factor = obj.unit.to(unit, equivalencies=equivalencies)
        return np.array([factor])

    # Determine the bunit "type". This will determine what information we need for the unit conversion.
    has_btemp = obj.unit.is_equivalent(u.K) or unit.is_equivalent(u.K)
    has_perbeam = obj.unit.is_equivalent(u.Jy/u.beam) or unit.is_equivalent(u.Jy/u.beam)
    has_perangarea = obj.unit.is_equivalent(u.Jy/u.sr) or unit.is_equivalent(u.Jy/u.sr)
    has_perpix = obj.unit.is_equivalent(u.Jy/u.pix) or unit.is_equivalent(u.Jy/u.pix)

    # Is there any beam object defined?
    has_beam = _has_beam(obj) or _has_beams(obj)

    # Set if this is a varying resolution object
    has_beams = _has_beams(obj)

    # Define freq, if needed:
    if any([has_perangarea, has_perbeam, has_btemp]):
        # Create a beam equivalency for brightness temperature
        # This requires knowing the frequency along the spectral axis.
        if freq is None:
            try:
                freq = obj.with_spectral_unit(u.Hz).spectral_axis
            except AttributeError:
                raise TypeError("Object of type {0} has no spectral "
                                "information. `freq` must be provided for"
                                " unit conversion from Jy/beam"
                                .format(type(obj)))
        else:
            if not freq.unit.is_equivalent(u.Hz):
                raise u.UnitsError("freq must be given in equivalent "
                                   "frequency units.")

            freq = freq.reshape((-1,))

    else:
        freq = [None]

    # To handle varying resolution objects, loop through "channels"
    # Default to a single iteration for a 2D spatial object or when a beam is not defined
    # This allows handling all 1D, 2D, and 3D data products.
    if has_beams:
        iter = range(len(obj.beams))
        beams = obj.beams
    elif has_beam:
        iter = range(0, 1)
        beams = [obj.beam]
    else:
        iter = range(0, 1)
        beams = [None]

    # Append the unit conversion factors
    factors = []

    # Iterate through spectral channels.
    for ii in iter:

        beam = beams[ii]

        # Use the range of frequencies when the beam does not change. Otherwise, select the
        # frequency corresponding to this beam.
        if has_beams:
            thisfreq = freq[ii]
        else:
            thisfreq = freq

        # Changes in beam require a new equivalency for each.
        this_equivalencies = deepcopy(equivalencies)

        # Equivalencies for Jy per ang area.
        if has_perangarea:
            bmequiv_angarea = u.brightness_temperature(thisfreq)

            this_equivalencies = list(this_equivalencies) + bmequiv_angarea

        # Beam area equivalencies for Jy per beam and/or Jy per ang area
        if has_perbeam:

            # create a beam equivalency for brightness temperature
            bmequiv = beam.jtok_equiv(thisfreq)

            # NOTE: `beamarea_equiv` was included in the radio-beam v0.3.3 release
            # The if/else here handles potential cases where earlier releases are installed.
            if hasattr(beam, 'beamarea_equiv'):
                bmarea_equiv = beam.beamarea_equiv
            else:
                bmarea_equiv = u.beam_angular_area(beam.sr)

            this_equivalencies = list(this_equivalencies) + bmequiv + bmarea_equiv

        # Equivalencies for Jy per pixel area.
        if has_perpix:

            if not obj.wcs.has_celestial:
                raise ValueError("Spatial WCS information is required for unit conversions"
                                " involving spatial areas (e.g., Jy/pix, Jy/sr)")

            pix_area = (proj_plane_pixel_area(obj.wcs.celestial) * u.deg**2).to(u.sr)

            pix_area_equiv = [(u.Jy / u.pix, u.Jy / u.sr,
                            lambda x: x / pix_area.value,
                            lambda x: x * pix_area.value)]

            this_equivalencies = list(this_equivalencies) + pix_area_equiv

            # Define full from brightness temp to Jy / pix.
            # Otherwise isn't working in 1 step
            if has_btemp:
                if not has_beam:
                    raise ValueError("Conversions between K and Jy/beam or Jy/pix"
                                    "requires the cube to have a beam defined.")

                jtok_factor = beam.jtok(thisfreq) / (u.Jy / u.beam)

                # We're going to do this piecemeal because it's easier to conceptualize
                # We specifically anchor these conversions based on the beam area. So from
                # beam to pix, this is beam -> angular area -> area per pixel
                # Altogether:
                # K ->  Jy/beam -> Jy /sr - > Jy / pix
                forward_factor = 1 / (jtok_factor * (beam.sr / u.beam) / (pix_area / u.pix))
                reverse_factor = jtok_factor * (beam.sr / u.beam) / (pix_area / u.pix)

                pix_area_btemp_equiv = [(u.K, u.Jy / u.pix,
                                        lambda x: x * forward_factor.value,
                                        lambda x: x * reverse_factor.value)]

                this_equivalencies = list(this_equivalencies) + pix_area_btemp_equiv

            # Equivalencies between pixel and angular areas.
            if has_perbeam:
                if not has_beam:
                    raise ValueError("Conversions between Jy/beam or Jy/pix"
                                    "requires the cube to have a beam defined.")

                beam_area = beam.sr

                pix_area_btemp_equiv = [(u.Jy / u.pix, u.Jy / u.beam,
                                        lambda x: x * (beam_area / pix_area).value,
                                        lambda x: x * (pix_area / beam_area).value)]

                this_equivalencies = list(this_equivalencies) + pix_area_btemp_equiv

        factor = obj.unit.to(unit, equivalencies=this_equivalencies)
        factors.append(factor)

    if has_beams:
        return factors
    else:
        # Slice along first axis to return a 1D array.
        return factors[0]

def combine_headers(header1, header2, **kwargs):
    '''
    Given two Header objects, this function returns a fits Header of the optimal wcs.

    Parameters
    ----------
    header1 : astropy.io.fits.Header
        A Header.
    header2 : astropy.io.fits.Header
        A Header.

    Returns
    -------
    header : astropy.io.fits.Header
        A header object of a field containing both initial headers.

    '''

    from reproject.mosaicking import find_optimal_celestial_wcs

    # Get wcs and shape of both headers
    w1 = WCS(header1).celestial
    s1 = w1.array_shape
    w2 = WCS(header2).celestial
    s2 = w2.array_shape

    # Get the optimal wcs and shape for both fields together
    wcs_opt, shape_opt = find_optimal_celestial_wcs([(s1, w1), (s2, w2)], auto_rotate=False,
                                                    **kwargs)

    # find spectral coverage
    specw1 = WCS(header1).spectral
    specw2 = WCS(header2).spectral
    specaxis1 = [x[0] for x in WCS(header1).world_axis_object_components].index('spectral')
    specaxis2 = [x[0] for x in WCS(header1).world_axis_object_components].index('spectral')
    range1 = specw1.pixel_to_world([0,header1[f'NAXIS{specaxis1+1}']-1])
    range2 = specw2.pixel_to_world([0,header2[f'NAXIS{specaxis1+1}']-1])

    # check for overlap
    # this will raise an exception if the headers are an different units, which we want
    if max(range1) < min(range2) or max(range2) < min(range1):
        warnings.warn(f"There is no spectral overlap between {range1} and {range2}")

    # check cdelt
    dx1 = specw1.proj_plane_pixel_scales()[0]
    dx2 = specw2.proj_plane_pixel_scales()[0]
    if dx1 != dx2:
        raise ValueError(f"Different spectral pixel scale {dx1} vs {dx2}")

    ranges = np.hstack([range1, range2])
    new_naxis = int(np.ceil((ranges.max() - ranges.min()) / np.abs(dx1)))

    # Make a new header using the optimal wcs and information from cubes
    header = header1.copy()
    header['NAXIS'] = 3
    header['NAXIS1'] = shape_opt[1]
    header['NAXIS2'] = shape_opt[0]
    header['NAXIS3'] = new_naxis
    header.update(wcs_opt.to_header())
    header['WCSAXES'] = 3
    return header

def _getdata(cube):
    """
    Must be defined out-of-scope to enable pickling
    """
    return (cube.unitless_filled_data[:], cube.wcs)

def mosaic_cubes(cubes, spectral_block_size=100, combine_header_kwargs={},
                 target_header=None,
                 commonbeam=None,
                 weightcubes=None,
                 save_to_tmp_dir=True,
                 use_memmap=True,
                 output_file=None,
                 method='cube',
                 verbose=True,
                 **kwargs):
    '''
    This function reprojects cubes onto a common grid and combines them to a single field.

    Parameters
    ----------
    cubes : iterable
        Iterable list of SpectralCube objects to reproject and add together.
    spectral_block_size : int
        Block size so that reproject does not run out of memory.
    combine_header_kwargs : dict
        Keywords passed to `~reproject.mosaicking.find_optimal_celestial_wcs`
        via `combine_headers`.
    commonbeam : Beam
        If specified, will smooth the data to this common beam before
        reprojecting.
    weightcubes : None
        Cubes with same shape as input cubes containing the weights
    save_to_tmp_dir : bool
        Default is to set `save_to_tmp_dir=True` because we expect cubes to be
        big.
    use_memmap : bool
        Use a memory-mapped array to save the mosaicked cube product?
    output_file : str or None
        If specified, this should be a FITS filename that the output *array*
        will be stored into (the footprint will not be saved)
    method : 'cube' or 'channel'
        Over what dimension should we iterate?  Options are 'cube' and
        'channel'.
    verbose : bool
        Progressbars?

    Outputs
    -------
    cube : SpectralCube
        A spectral cube with the list of cubes mosaicked together.
    '''
    from reproject.mosaicking import reproject_and_coadd
    from reproject import reproject_interp
    import warnings
    from astropy.utils.exceptions import AstropyUserWarning
    warnings.filterwarnings('ignore', category=AstropyUserWarning)

    if verbose:
        from tqdm import tqdm as std_tqdm
    else:
        class tqdm:
            def __init__(self, x):
                return x
            def __call__(self, x):
                return x
            def set_description(self, **kwargs):
                pass
            def update(self, **kwargs):
                pass
        std_tqdm = tqdm

    def log_(x):
        if verbose:
            log.info(x)
        else:
            log.debug(x)

    cube1 = cubes[0]

    if target_header is None:
        target_header = cube1.header

        # Create a header for a field containing all cubes
        for cu in cubes[1:]:
            target_header = combine_headers(target_header, cu.header, **combine_header_kwargs)

    # Prepare an array and mask for the final cube
    shape_opt = (target_header['NAXIS3'], target_header['NAXIS2'], target_header['NAXIS1'])
    dtype = f"float{int(abs(target_header['BITPIX']))}"

    if output_file is not None:
        log_(f"Using output file {output_file}")
        t0 = time.time()
        if not output_file.endswith('.fits'):
            raise IOError("Only FITS output is supported")
        if not os.path.exists(output_file):
            # https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html#sphx-glr-generated-examples-io-skip-create-large-fits-py
            hdu = fits.PrimaryHDU(data=np.ones([5,5,5], dtype=dtype),
                                  header=target_header
                                 )
            for kwd in ('NAXIS1', 'NAXIS2', 'NAXIS3'):
                hdu.header[kwd] = target_header[kwd]

            log_(f"Dumping header to file {output_file} (dt={time.time()-t0})")
            target_header.tofile(output_file, overwrite=True)
            with open(output_file, 'rb+') as fobj:
                fobj.seek(len(target_header.tostring()) +
                          (np.prod(shape_opt) * np.abs(target_header['BITPIX']//8)) - 1)
                fobj.write(b'\0')

        log_(f"Loading header from file {output_file} (dt={time.time()-t0})")
        hdu = fits.open(output_file, mode='update', overwrite=True)
        output_array = hdu[0].data
        hdu.flush() # make sure the header gets written right

        log_(f"Creating footprint file dt={time.time()-t0}")
        # use memmap - not a FITS file - for the footprint
        # if we want to do partial writing, though, it's best to make footprint an extension
        ntf2 = tempfile.NamedTemporaryFile()
        output_footprint = np.memmap(ntf2, mode='w+', shape=shape_opt, dtype=dtype)

        # default footprint to 1 assuming there is some stuff already in the image
        # this is a hack and maybe just shouldn't be attempted
        #log_("Initializing footprint to 1s")
        #output_footprint[:] = 1 # takes an hour?!
        log_(f"Done initializing memory dt={time.time()-t0}")
    elif use_memmap:
        log_("Using memmap")
        ntf = tempfile.NamedTemporaryFile()
        output_array = np.memmap(ntf, mode='w+', shape=shape_opt, dtype=dtype)
        ntf2 = tempfile.NamedTemporaryFile()
        output_footprint = np.memmap(ntf2, mode='w+', shape=shape_opt, dtype=dtype)
    else:
        log_("Using memory")
        output_array = np.zeros(shape_opt)
        output_footprint = np.zeros(shape_opt)
    mask_opt = np.zeros(shape_opt[1:])


    # check that the beams are deconvolvable
    if commonbeam is not None:
        # assemble beams
        beams = [cube.beam if hasattr(cube, 'beam') else cube.beams.common_beam()
                 for cube in cubes]

        for beam in beams:
            # this will raise an exception if any of the cubes have bad beams
            commonbeam.deconvolve(beam)

    if verbose:
        class tqdm(std_tqdm):
            def update(self, n=1):
                hdu.flush() # write to disk on each iteration
                super().update(n)

    if method == 'cube':
        log_("Using Cube method")
        # Cube method: Regrid the whole cube in one operation.
        # Let reproject_and_coadd handle any iterations

        if commonbeam is not None:
            cubes = [cube.convolve_to(commonbeam, save_to_tmp_dir=save_to_tmp_dir)
                     for cube in std_tqdm(cubes, desc="Convolve:")]

        try:
            output_array, output_footprint = reproject_and_coadd(
                [cube.hdu for cube in cubes],
                target_header,
                input_weights=[cube.hdu for cube in weightcubes] if weightcubes is None else None,
                output_array=output_array,
                output_footprint=output_footprint,
                reproject_function=reproject_interp,
                progressbar=tqdm if verbose else False,
                block_size=(None if spectral_block_size is None else
                            [(spectral_block_size, cube.shape[1], cube.shape[2])
                             for cube in cubes]),
            )
        except TypeError as ex:
            # print the exception in case we caught a different TypeError than expected
            warnings.warn("The block_size argument is not accepted by `reproject`.  "
                          f"A more recent version may be needed.  Exception was: {ex}")
            output_array, output_footprint = reproject_and_coadd(
                [cube.hdu for cube in cubes],
                target_header,
                input_weights=[cube.hdu for cube in weightcubes] if weightcubes is None else None,
                output_array=output_array,
                output_footprint=output_footprint,
                reproject_function=reproject_interp,
                progressbar=tqdm if verbose else False,
            )
    elif method == 'channel':
        log_("Using Channel method")
        # Channel method: manually downselect to go channel-by-channel in the
        # input cubes before handing off material to reproject_and_coadd This
        # approach allows us more direct & granular control over memory and is
        # likely better for large-area cubes
        # (ideally we'd let Dask handle all the memory allocation choices under
        # the hood, but as of early 2023, we do not yet have that capability)

        outwcs = WCS(target_header)
        channels = outwcs.spectral.pixel_to_world(np.arange(target_header['NAXIS3']))
        dx = outwcs.spectral.proj_plane_pixel_scales()[0]
        log_(f"Channel mode: dx={dx}.  Looping over {len(channels)} channels and {len(cubes)} cubes")

        mincube_slices = [cube[cube.shape[0]//2:cube.shape[0]//2+1]
                          .subcube_slices_from_mask(cube[cube.shape[0]//2:cube.shape[0]//2+1].mask,
                                                    spatial_only=True)
                          for cube in std_tqdm(cubes, desc='MinSubSlices:', delay=5)]

        pbar = tqdm(enumerate(channels), desc="Channels")
        for ii, channel in pbar:
            pbar.set_description(f"Channel {ii}={channel}")

            # grab a 2-channel slab
            # this is very verbose but quite simple & cheap
            # going to spectral_slab(channel-dx, channel+dx) gives 3-pixel cubes most often,
            # which results in a 50% overhead in smoothing, etc.
            chans = [(cube.closest_spectral_channel(channel) if cube.spectral_axis[cube.closest_spectral_channel(channel)] < channel else cube.closest_spectral_channel(channel)+1,
                      cube.closest_spectral_channel(channel) if cube.spectral_axis[cube.closest_spectral_channel(channel)] > channel else cube.closest_spectral_channel(channel)-1)
                     for cube in std_tqdm(cubes, delay=5, desc='ChanSel:')]
            # reversed spectral axes still break things
            # and we want two channels width, not one
            chans = [(ch1, ch2+1) if ch1 < ch2 else (ch2, ch1+1) for ch1, ch2 in chans]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') # seriously NO WARNINGS.

                scubes = [(cube[ch1:ch2, slices[1], slices[2]]
                           .convolve_to(commonbeam)
                           .rechunk())
                          for (ch1, ch2), slices, cube in std_tqdm(zip(chans, mincube_slices, cubes),
                                                                   delay=5, desc='Subcubes')]
                # only keep cubes that are in range; the rest get excluded
                keep = [(cube.shape[0] > 1) and
                        (cube.spectral_axis.min() < channel) and
                        (cube.spectral_axis.max() > channel)
                        for cube in scubes]
                if sum(keep) < len(keep):
                    log.warn(f"Dropping {len(keep)-sum(keep)} cubes out of {len(keep)} because they're out of range")
                    scubes = [cube for cube, kp in zip(scubes, keep) if kp]

                if weightcubes is not None:
                    sweightcubes = [cube[ch1:ch2, slices[1], slices[2]]
                                    for (ch1, ch2), slices, cube, kp
                                    in std_tqdm(zip(chans, mincube_slices, weightcubes, keep),
                                                delay=5, desc='Subweight')
                                    if kp
                                    ]
                    wthdus = [cube.hdu
                              for cube in std_tqdm(sweightcubes, delay=5, desc='WeightData')]


                # reproject_and_coadd requires the actual arrays, so this is the convolution step

                # commented out approach here: just let spectral-cube handle the convolution etc.
                #hdus = [(cube._get_filled_data(), cube.wcs)
                #        for cube in std_tqdm(scubes, delay=5, desc='Data/conv')]

                # somewhat faster (?) version - ask the dask client to handle
                # gathering the data
                # (this version is capable of parallelizing over many cubes, in
                # theory; the previous would treat each cube in serial)
                datas = [cube._get_filled_data() for cube in scubes]
                wcses = [cube.wcs for cube in scubes]
                with Client() as client:
                    datas = client.gather(datas)
                hdus = list(zip(datas, wcses))

                # project into array w/"dummy" third dimension
                # (outputs are not used; data is written directly into the output array chunks)
                output_array_, output_footprint_ = reproject_and_coadd(
                    hdus,
                    outwcs[ii:ii+1, :, :],
                    shape_out=(1,) + output_array.shape[1:],
                    output_array=output_array[ii:ii+1,:,:],
                    output_footprint=output_footprint[ii:ii+1,:,:],
                    reproject_function=reproject_interp,
                    input_weights=wthdus,
                    progressbar=partial(tqdm, desc='coadd') if verbose else False,
                )

            pbar.set_description(f"Channel {ii}={channel} done")

    # Create Cube
    cube = cube1.__class__(data=output_array * cube1.unit, wcs=WCS(target_header))

    if output_file is not None:
        hdu.flush()
        hdu.close()

    return cube
