from __future__ import print_function, absolute_import, division

import contextlib
import warnings
from copy import deepcopy

try:
    import builtins
except ImportError:
    # python2
    import __builtin__ as builtins

import dask.array as da
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

    # Make a new header using the optimal wcs and information from cubes
    header = header1.copy()
    header['NAXIS'] = 3
    header['NAXIS1'] = shape_opt[1]
    header['NAXIS2'] = shape_opt[0]
    header['NAXIS3'] = header1['NAXIS3']
    header.update(wcs_opt.to_header())
    header['WCSAXES'] = 3
    return header

def mosaic_cubes(cubes, spectral_block_size=100, combine_header_kwargs={}, **kwargs):
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
    Outputs
    -------
    cube : SpectralCube
        A spectral cube with the list of cubes mosaicked together.
    '''

    cube1 = cubes[0]
    header = cube1.header

    # Create a header for a field containing all cubes
    for cu in cubes[1:]:
        header = combine_headers(header, cu.header, **combine_header_kwargs)

    # Prepare an array and mask for the final cube
    shape_opt = (header['NAXIS3'], header['NAXIS2'], header['NAXIS1'])
    final_array = np.zeros(shape_opt)
    mask_opt = np.zeros(shape_opt[1:])

    for cube in cubes:
        # Reproject cubes to the header
        try:
            if spectral_block_size is not None:
                cube_repr = cube.reproject(header,
                                           block_size=[spectral_block_size,
                                                       cube.shape[1],
                                                       cube.shape[2]],
                                           **kwargs)
            else:
                cube_repr = cube.reproject(header, **kwargs)
        except TypeError:
            warnings.warn("The block_size argument is not accepted by `reproject`.  "
                          "A more recent version may be needed.")
            cube_repr = cube.reproject(header, **kwargs)

        # Create weighting mask (2D)
        mask = (cube_repr[0:1].get_mask_array()[0])
        mask_opt += mask.astype(float)

        # Go through each slice of the cube, add it to the final array
        for ii in range(final_array.shape[0]):
            slice1 = np.nan_to_num(cube_repr.unitless_filled_data[ii])
            final_array[ii] = final_array[ii] + slice1

    # Dividing by the mask throws errors where it is zero
    with np.errstate(divide='ignore'):

        # Use weighting mask to average where cubes overlap
        for ss in range(final_array.shape[0]):
            final_array[ss] /= mask_opt

    # Create Cube
    cube = cube1.__class__(data=final_array * cube1.unit, wcs=WCS(header))
    return cube
