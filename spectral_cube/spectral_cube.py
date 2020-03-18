"""
A class to represent a 3-d position-position-velocity spectral cube.
"""

from __future__ import print_function, absolute_import, division

import warnings
from functools import wraps
import operator
import re
import itertools
import copy
import tempfile
import textwrap
from pathlib import PosixPath
import six
from six.moves import zip, range
import dask.array

import astropy.wcs
from astropy import units as u
from astropy.io.fits import PrimaryHDU, BinTableHDU, Header, Card, HDUList
from astropy.utils.console import ProgressBar
from astropy import log
from astropy import wcs
from astropy import convolution
from astropy import stats
from astropy.constants import si
from astropy.io.registry import UnifiedReadWriteMethod

import numpy as np

from radio_beam import Beam, Beams

from . import cube_utils
from . import wcs_utils
from . import spectral_axis
from .masks import (LazyMask, LazyComparisonMask, BooleanArrayMask, MaskBase,
                    is_broadcastable_and_smaller)
from .ytcube import ytCube
from .lower_dimensional_structures import (Projection, Slice, OneDSpectrum,
                                           LowerDimensionalObject,
                                           VaryingResolutionOneDSpectrum
                                          )
from .base_class import (BaseNDClass, SpectralAxisMixinClass,
                         DOPPLER_CONVENTIONS, SpatialCoordMixinClass,
                         MaskableArrayMixinClass, MultiBeamMixinClass,
                         HeaderMixinClass, BeamMixinClass,
                        )
from .utils import (cached, warn_slow, VarianceWarning, BeamWarning,
                    UnsupportedIterationStrategyWarning, WCSMismatchWarning,
                    NotImplementedWarning, SliceWarning, SmoothingWarning,
                    StokesWarning, ExperimentalImplementationWarning,
                    BeamAverageWarning, NonFiniteBeamsWarning, BeamWarning)
from .spectral_axis import (determine_vconv_from_ctype, get_rest_value_from_wcs,
                            doppler_beta, doppler_gamma, doppler_z)
from .io.core import SpectralCubeRead, SpectralCubeWrite

from distutils.version import LooseVersion


__all__ = ['BaseSpectralCube', 'SpectralCube', 'VaryingResolutionSpectralCube']

# apply_everywhere, world: do not have a valid cube to test on
__doctest_skip__ = ['BaseSpectralCube._apply_everywhere']

try:
    from scipy import ndimage
    scipyOK = True
except ImportError:
    scipyOK = False

warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning, append=True)

SIGMA2FWHM = 2. * np.sqrt(2. * np.log(2.))

# convenience structures to keep track of the reversed index
# conventions between WCS and numpy
np2wcs = {2: 0, 1: 1, 0: 2}


_NP_DOC = """
Ignores excluded mask elements.

Parameters
----------
axis : int (optional)
   The axis to collapse, or None to perform a global aggregation
how : cube | slice | ray | auto
   How to compute the aggregation. All strategies give the same
   result, but certain strategies are more efficient depending
   on data size and layout. Cube/slice/ray iterate over
   decreasing subsets of the data, to conserve memory.
   Default='auto'
""".replace('\n', '\n        ')


def aggregation_docstring(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__doc__ += _NP_DOC
    return wrapper


_PARALLEL_DOC = """

Other Parameters
----------------
parallel : bool
    Use joblib to parallelize the operation.
    If set to ``False``, will force the use of a single core without
    using ``joblib``.
num_cores : int or None
    The number of cores to use when applying this function in parallel
    across the cube.
use_memmap : bool
    If specified, a memory mapped temporary file on disk will be
    written to rather than storing the intermediate spectra in memory.
"""


def parallel_docstring(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    line1 = wrapper.__doc__.split("\n")[1]
    indentation = " "*(len(line1) - len(line1.lstrip()))

    try:
        wrapper.__doc__ += textwrap.indent(_PARALLEL_DOC, indentation)
    except AttributeError:
        # python2.7
        wrapper.__doc__ = textwrap.dedent(wrapper.__doc__) + _PARALLEL_DOC

    return wrapper

def _apply_spectral_function(arguments, outcube, function, shape=None,
                             return_vals=False, **kwargs):
    """
    Helper function to apply a function to a spectrum.
    Needs to be declared toward the top of the code to allow pickling by
    joblib.
    """
    (spec, includemask, ii, jj) = arguments
    try:
        spec = spec(ii, jj)
        includemask = includemask(ii, jj)
    except TypeError:
        # assume already arrays
        pass

    if isinstance(outcube, str):
        outcube = DelayedMemmapWriter(filename=outcube, shape=shape,
                                      dtype=np.float)

    if np.any(includemask):
        outcube[:, jj, ii] = function(spec, **kwargs)
    else:              
        outcube[:, jj, ii] = spec

    if return_vals:
        return outcube[:, jj, ii]

def _apply_spatial_function(arguments, outcube, function, shape=None,
                            return_vals=False, **kwargs):
    """
    Helper function to apply a function to an image.
    Needs to be declared toward the top of the code to allow pickling by
    joblib.
    """
    (img, includemask, ii) = arguments
    try:
        img = img(ii)
        includemask = includemask(ii)
    except TypeError:
        # assume already arrays
        pass

    if isinstance(outcube, str):
        outcube = DelayedMemmapWriter(filename=outcube, shape=shape,
                                      dtype=np.float)

    if np.any(includemask):
        outcube[ii, :, :] = function(img, **kwargs)
    else:
        outcube[ii, :, :] = img

    if return_vals:
        return outcube[ii, :, :]

class DelayedMemmapWriter(object):
    """
    A peculiar object that wraps memmap objects that already exist to allow
    independent threads to write to them
    """
    def __init__(self, filename, shape, dtype):
        self.filename = filename
        self.shape = shape
        self.dtype = dtype

    def __setitem__(self, view, value):
        out = np.memmap(filename=self.filename, mode='r+', shape=self.shape,
                        dtype=self.dtype)
        out[view] = value
        out.flush()

class BaseSpectralCube(BaseNDClass, MaskableArrayMixinClass,
                       SpectralAxisMixinClass, SpatialCoordMixinClass,
                       HeaderMixinClass):

    def __init__(self, data, wcs, mask=None, meta=None, fill_value=np.nan,
                 header=None, allow_huge_operations=False, wcs_tolerance=0.0):

        # Deal with metadata first because it can affect data reading
        self._meta = meta or {}

        # must extract unit from data before stripping it
        if 'BUNIT' in self._meta:
            self._unit = cube_utils.convert_bunit(self._meta["BUNIT"])
        elif hasattr(data, 'unit'):
            self._unit = data.unit
        else:
            self._unit = None

        # data must not be a quantity when stored in self._data
        if hasattr(data, 'unit'):
            # strip the unit so that it can be treated as cube metadata
            data = data.value

        # TODO: mask should be oriented? Or should we assume correctly oriented here?
        self._data, self._wcs = cube_utils._orient(data, wcs)
        self._wcs_tolerance = wcs_tolerance
        self._spectral_axis = None
        self._mask = mask  # specifies which elements to Nan/blank/ignore
                           # object or array-like object, given that WCS needs
                           # to be consistent with data?
        #assert mask._wcs == self._wcs
        self._fill_value = fill_value

        self._header = Header() if header is None else header
        if not isinstance(self._header, Header):
            raise TypeError("If a header is given, it must be a fits.Header")

        # We don't pass the spectral unit via the initializer since the user
        # should be using ``with_spectral_unit`` if they want to set it.
        # However, we do want to keep track of what units the spectral axis
        # should be returned in, otherwise astropy's WCS can change the units,
        # e.g. km/s -> m/s.
        # This can be overridden with Header below
        self._spectral_unit = u.Unit(self._wcs.wcs.cunit[2])

        # This operation is kind of expensive?
        header_specaxnum = astropy.wcs.WCS(header).wcs.spec
        header_specaxunit = spectral_axis.unit_from_header(self._header,
                                                           spectral_axis_number=header_specaxnum+1)

        # Allow the original header spectral axis unit to override the default
        # unit
        if header_specaxunit is not None:
            self._spectral_unit = header_specaxunit

        self._spectral_scale = spectral_axis.wcs_unit_scale(self._spectral_unit)

        self.allow_huge_operations = allow_huge_operations

        self._cache = {}

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def _is_huge(self):
        return cube_utils.is_huge(self)

    @property
    def _new_thing_with(self):
        return self._new_cube_with

    def _new_cube_with(self, data=None, wcs=None, mask=None, meta=None,
                       fill_value=None, spectral_unit=None, unit=None,
                       wcs_tolerance=None, **kwargs):

        data = self._data if data is None else data
        if unit is None and hasattr(data, 'unit'):
            if data.unit != self.unit:
                raise u.UnitsError("New data unit '{0}' does not"
                                   " match cube unit '{1}'.  You can"
                                   " override this by specifying the"
                                   " `unit` keyword."
                                   .format(data.unit, self.unit))
            unit = data.unit
        elif unit is not None:
            # convert string units to Units
            if not isinstance(unit, u.Unit):
                unit = u.Unit(unit)

            if hasattr(data, 'unit'):
                if u.Unit(unit) != data.unit:
                    raise u.UnitsError("The specified new cube unit '{0}' "
                                       "does not match the input unit '{1}'."
                                       .format(unit, data.unit))
            else:
                data = u.Quantity(data, unit=unit, copy=False)
        elif self._unit is not None:
            unit = self.unit

        wcs = self._wcs if wcs is None else wcs
        mask = self._mask if mask is None else mask
        if meta is None:
            meta = {}
            meta.update(self._meta)
        if unit is not None:
            meta['BUNIT'] = unit.to_string(format='FITS')

        fill_value = self._fill_value if fill_value is None else fill_value
        spectral_unit = self._spectral_unit if spectral_unit is None else u.Unit(spectral_unit)

        cube = self.__class__(data=data, wcs=wcs, mask=mask, meta=meta,
                              fill_value=fill_value, header=self._header,
                              allow_huge_operations=self.allow_huge_operations,
                              wcs_tolerance=wcs_tolerance or self._wcs_tolerance,
                              **kwargs)
        cube._spectral_unit = spectral_unit
        cube._spectral_scale = spectral_axis.wcs_unit_scale(spectral_unit)

        return cube

    read = UnifiedReadWriteMethod(SpectralCubeRead)
    write = UnifiedReadWriteMethod(SpectralCubeWrite)

    @property
    def unit(self):
        """ The flux unit """
        if self._unit:
            return self._unit
        else:
            return u.one

    @property
    def shape(self):
        """ Length of cube along each axis """
        return self._data.shape

    @property
    def size(self):
        """ Number of elements in the cube """
        return self._data.size

    @property
    def base(self):
        """ The data type 'base' of the cube - useful for, e.g., joblib """
        return self._data.base

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        """ Dimensionality of the data """
        return self._data.ndim

    def __repr__(self):
        s = "{1} with shape={0}".format(self.shape, self.__class__.__name__)
        if self.unit is u.one:
            s += ":\n"
        else:
            s += " and unit={0}:\n".format(self.unit)
        s += (" n_x: {0:6d}  type_x: {1:8s}  unit_x: {2:5s}"
              "  range: {3:12.6f}:{4:12.6f}\n".format(self.shape[2],
                                                      self.wcs.wcs.ctype[0],
                                                      self.wcs.wcs.cunit[0],
                                                      self.longitude_extrema[0],
                                                      self.longitude_extrema[1],))
        s += (" n_y: {0:6d}  type_y: {1:8s}  unit_y: {2:5s}"
              "  range: {3:12.6f}:{4:12.6f}\n".format(self.shape[1],
                                                      self.wcs.wcs.ctype[1],
                                                      self.wcs.wcs.cunit[1],
                                                      self.latitude_extrema[0],
                                                      self.latitude_extrema[1],
                                                     ))
        s += (" n_s: {0:6d}  type_s: {1:8s}  unit_s: {2:5s}"
              "  range: {3:12.3f}:{4:12.3f}".format(self.shape[0],
                                                    self.wcs.wcs.ctype[2],
                                                    self._spectral_unit,
                                                    self.spectral_extrema[0],
                                                    self.spectral_extrema[1],
                                                   ))
        return s

    @property
    @cached
    def spectral_extrema(self):
        _spectral_min = self.spectral_axis.min()
        _spectral_max = self.spectral_axis.max()

        return _spectral_min, _spectral_max

    def apply_numpy_function(self, function, fill=np.nan,
                             reduce=True, how='auto',
                             projection=False,
                             unit=None,
                             check_endian=False,
                             progressbar=False,
                             includemask=False,
                             **kwargs):
        """
        Apply a numpy function to the cube

        Parameters
        ----------
        function : Numpy ufunc
            A numpy ufunc to apply to the cube
        fill : float
            The fill value to use on the data
        reduce : bool
            reduce indicates whether this is a reduce-like operation,
            that can be accumulated one slice at a time.
            sum/max/min are like this. argmax/argmin/stddev are not
        how : cube | slice | ray | auto
           How to compute the moment. All strategies give the same
           result, but certain strategies are more efficient depending
           on data size and layout. Cube/slice/ray iterate over
           decreasing subsets of the data, to conserve memory.
           Default='auto'
        projection : bool
            Return a :class:`~spectral_cube.lower_dimensional_structures.Projection` if the resulting array is 2D or a
            OneDProjection if the resulting array is 1D and the sum is over both
            spatial axes?
        unit : None or `astropy.units.Unit`
            The unit to include for the output array.  For example,
            `SpectralCube.max` calls
            ``SpectralCube.apply_numpy_function(np.max, unit=self.unit)``,
            inheriting the unit from the original cube.
            However, for other numpy functions, e.g. `numpy.argmax`, the return
            is an index and therefore unitless.
        check_endian : bool
            A flag to check the endianness of the data before applying the
            function.  This is only needed for optimized functions, e.g. those
            in the `bottleneck <https://pypi.python.org/pypi/Bottleneck>`_ package.
        progressbar : bool
            Show a progressbar while iterating over the slices through the
            cube?
        kwargs : dict
            Passed to the numpy function.

        Returns
        -------
        result : :class:`~spectral_cube.lower_dimensional_structures.Projection` or `~astropy.units.Quantity` or float
            The result depends on the value of ``axis``, ``projection``, and
            ``unit``.  If ``axis`` is None, the return will be a scalar with or
            without units.  If axis is an integer, the return will be a
            :class:`~spectral_cube.lower_dimensional_structures.Projection` if ``projection`` is set
        """


        # leave axis in kwargs to avoid overriding numpy defaults, e.g.  if the
        # default is axis=-1, we don't want to force it to be axis=None by
        # specifying that in the function definition
        axis = kwargs.get('axis', None)

        if how == 'auto':
            strategy = cube_utils.iterator_strategy(self, axis)
        else:
            strategy = how

        out = None

        log.debug("applying numpy function {0} with strategy {1}"
                  .format(function, strategy))

        if strategy == 'slice' and reduce:
            out = self._reduce_slicewise(function, fill, check_endian,
                                         includemask=includemask,
                                         progressbar=progressbar, **kwargs)
        elif how == 'ray':
            out = self.apply_function(function, **kwargs)
        elif how not in ['auto', 'cube']:
            warnings.warn("Cannot use how=%s. Using how=cube" % how,
                          UnsupportedIterationStrategyWarning)

        if out is None:
            out = function(self._get_filled_data(fill=fill,
                                                 check_endian=check_endian),
                           **kwargs)

        if axis is None:
            # return is scalar
            if unit is not None:
                return u.Quantity(out, unit=unit)
            else:
                return out
        elif projection and reduce:
            meta = {'collapse_axis': axis}
            meta.update(self._meta)

            if hasattr(axis, '__len__') and len(axis) == 2:
                # if operation is over two spatial dims
                if set(axis) == set((1,2)):
                    new_wcs = self._wcs.sub([wcs.WCSSUB_SPECTRAL])
                    header = self._nowcs_header
                    # check whether the cube has beams at all
                    # (note that "hasattr(self, 'beam') on an object with no
                    # _beam will result in an exception....?!?!?!?)
                    if hasattr(self, '_beam') and self._beam is not None:
                        bmarg = {'beam': self.beam}
                    elif hasattr(self, '_beams') and self._beams is not None:
                        bmarg = {'beams': self.unmasked_beams}
                    else:
                        bmarg = {}
                    return self._oned_spectrum(value=out,
                                               wcs=new_wcs,
                                               copy=False,
                                               unit=unit,
                                               header=header,
                                               meta=meta,
                                               spectral_unit=self._spectral_unit,
                                               **bmarg
                                              )
                else:
                    warnings.warn("Averaging over a spatial and a spectral "
                                  "dimension cannot produce a Projection "
                                  "quantity (no units or WCS are preserved).",
                                  SliceWarning
                                 )
                    return out

            else:
                new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])
                header = self._nowcs_header

                return Projection(out, copy=False, wcs=new_wcs, meta=meta,
                                  unit=unit, header=header)
        else:
            return out

    def _reformat_cube_output(self, out, axis, unit, projection=False,
                             reduce=True):
        """
        Helper function for all the various parallel things to handle
        the re-formation of the output object
        """
        if axis is None and reduce:
            # return is scalar
            if unit is not None:
                return u.Quantity(out, unit=unit)
            else:
                return out
        elif projection and reduce:
            meta = {'collapse_axis': axis}
            meta.update(self._meta)

            if hasattr(axis, '__len__') and len(axis) == 2:
                # if operation is over two spatial dims
                if set(axis) == set((1,2)):
                    new_wcs = self._wcs.sub([wcs.WCSSUB_SPECTRAL])
                    header = self._nowcs_header
                    if hasattr(self, 'beam'):
                        bmarg = {'beam': self.beam}
                    elif hasattr(self, 'beams'):
                        bmarg = {'beams': self.unmasked_beams}
                    else:
                        bmarg = {}
                    return self._oned_spectrum(value=out,
                                               wcs=new_wcs,
                                               copy=False,
                                               unit=unit,
                                               header=header,
                                               meta=meta,
                                               spectral_unit=self._spectral_unit,
                                               **bmarg
                                              )
                else:
                    warnings.warn("Averaging over a spatial and a spectral "
                                  "dimension cannot produce a Projection "
                                  "quantity (no units or WCS are preserved).",
                                  SliceWarning
                                 )
                    return out

            else:
                new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])
                header = self._nowcs_header

                return Projection(out, copy=False, wcs=new_wcs, meta=meta,
                                  unit=unit, header=header)
        else:
            # if the return is a masked array, parse that "correctly"
            if hasattr(out, 'mask'):
                new_mask = self.mask & BooleanArrayMask(out.mask, wcs=self.wcs)
                out = out.data
            newcube = self._new_cube_with(data=out, wcs=self.wcs,
                                          mask=self.mask, meta=self.meta,
                                          fill_value=self.fill_value)
            return newcube


    def dask_apply_along_axis(self, function, axis, fill=np.nan, reduce=True,
                              projection=False, unit=None, check_endian=False,
                              includemask=False,
                              **kwargs):
        """
        Apply a numpy function to the cube

        Parameters
        ----------
        function : a function
            A function to apply to the cube
        fill : float
            The fill value to use on the data
        reduce : bool
            reduce indicates whether this is a reduce-like operation,
            that can be accumulated one slice at a time.
            sum/max/min are like this. argmax/argmin/stddev are not
        how : cube | slice | ray | auto
           How to compute the moment. All strategies give the same
           result, but certain strategies are more efficient depending
           on data size and layout. Cube/slice/ray iterate over
           decreasing subsets of the data, to conserve memory.
           Default='auto'
        projection : bool
            Return a :class:`~spectral_cube.lower_dimensional_structures.Projection` if the resulting array is 2D or a
            OneDProjection if the resulting array is 1D and the sum is over both
            spatial axes?
        unit : None or `astropy.units.Unit`
            The unit to include for the output array.  For example,
            `SpectralCube.max` calls
            ``SpectralCube.apply_numpy_function(np.max, unit=self.unit)``,
            inheriting the unit from the original cube.
            However, for other numpy functions, e.g. `numpy.argmax`, the return
            is an index and therefore unitless.
        check_endian : bool
            A flag to check the endianness of the data before applying the
            function.  This is only needed for optimized functions, e.g. those
            in the `bottleneck <https://pypi.python.org/pypi/Bottleneck>`_ package.
        kwargs : dict
            Passed to the numpy function.

        Returns
        -------
        result : :class:`~spectral_cube.lower_dimensional_structures.Projection` or `~astropy.units.Quantity` or float
            The result depends on the value of ``axis``, ``projection``, and
            ``unit``.  If ``axis`` is None, the return will be a scalar with or
            without units.  If axis is an integer, the return will be a
            :class:`~spectral_cube.lower_dimensional_structures.Projection` if ``projection`` is set
        """


        out = None

        log.debug("applying dask function {0}".format(function))

        import dask.array

        da = dask.array.from_array(self.unitless_filled_data, chunks="auto")

        # might consider more arguments here, even rechunking depending on the
        # axis?
        operation = dask.array.apply_along_axis(function, axis=axis,
                                                arr=da, **kwargs)

        out = operation.compute()

        return self._reformat_cube_output(out=out, axis=axis, unit=unit)


    def dask_apply_function(self, function, fill=np.nan, reduce=True,
                            projection=False, unit=None, check_endian=False,
                            includemask=False, **kwargs):
        """
        Apply a numpy function to the cube

        Parameters
        ----------
        function : a function
            A function to apply to the cube
        fill : float
            The fill value to use on the data
        reduce : bool
            reduce indicates whether this is a reduce-like operation,
            that can be accumulated one slice at a time.
            sum/max/min are like this. argmax/argmin/stddev are not
        how : cube | slice | ray | auto
           How to compute the moment. All strategies give the same
           result, but certain strategies are more efficient depending
           on data size and layout. Cube/slice/ray iterate over
           decreasing subsets of the data, to conserve memory.
           Default='auto'
        projection : bool
            Return a :class:`~spectral_cube.lower_dimensional_structures.Projection` if the resulting array is 2D or a
            OneDProjection if the resulting array is 1D and the sum is over both
            spatial axes?
        unit : None or `astropy.units.Unit`
            The unit to include for the output array.  For example,
            `SpectralCube.max` calls
            ``SpectralCube.apply_numpy_function(np.max, unit=self.unit)``,
            inheriting the unit from the original cube.
            However, for other numpy functions, e.g. `numpy.argmax`, the return
            is an index and therefore unitless.
        check_endian : bool
            A flag to check the endianness of the data before applying the
            function.  This is only needed for optimized functions, e.g. those
            in the `bottleneck <https://pypi.python.org/pypi/Bottleneck>`_ package.
        kwargs : dict
            Passed to the numpy function.

        Returns
        -------
        result : :class:`~spectral_cube.lower_dimensional_structures.Projection` or `~astropy.units.Quantity` or float
            The result depends on the value of ``axis``, ``projection``, and
            ``unit``.  If ``axis`` is None, the return will be a scalar with or
            without units.  If axis is an integer, the return will be a
            :class:`~spectral_cube.lower_dimensional_structures.Projection` if ``projection`` is set
        """


        # leave axis in kwargs to avoid overriding numpy defaults, e.g.  if the
        # default is axis=-1, we don't want to force it to be axis=None by
        # specifying that in the function definition
        axis = kwargs.get('axis', None)

        out = None

        log.debug("applying dask function {0}".format(function))

        import dask.array

        da = dask.array.from_array(self.unitless_filled_data, chunks="auto")

        # might consider more arguments here, even rechunking depending on the
        # axis?
        operation = da.map_blocks(function,
                                  drop_axis=axis if reduce else None,
                                  **kwargs
                                 )

        out = operation.compute()

        return self._reformat_cube_output(out, axis, unit)

    def dask_apply_function_by_image(self, function, fill=np.nan, reduce=False,
                                     projection=False, unit=None,
                                     check_endian=False,
                                     use_memmap=True,
                                     memmap_dir=None,
                                     **kwargs):
        """
        Apply a numpy function to the cube

        Parameters
        ----------
        function : a function
            A function to apply to the cube
        fill : float
            The fill value to use on the data
        reduce : bool
            reduce indicates whether this is a reduce-like operation,
            that can be accumulated one slice at a time.
            sum/max/min are like this. argmax/argmin/stddev are not
        how : cube | slice | ray | auto
           How to compute the moment. All strategies give the same
           result, but certain strategies are more efficient depending
           on data size and layout. Cube/slice/ray iterate over
           decreasing subsets of the data, to conserve memory.
           Default='auto'
        projection : bool
            Return a :class:`~spectral_cube.lower_dimensional_structures.Projection` if the resulting array is 2D or a
            OneDProjection if the resulting array is 1D and the sum is over both
            spatial axes?
        unit : None or `astropy.units.Unit`
            The unit to include for the output array.  For example,
            `SpectralCube.max` calls
            ``SpectralCube.apply_numpy_function(np.max, unit=self.unit)``,
            inheriting the unit from the original cube.
            However, for other numpy functions, e.g. `numpy.argmax`, the return
            is an index and therefore unitless.
        check_endian : bool
            A flag to check the endianness of the data before applying the
            function.  This is only needed for optimized functions, e.g. those
            in the `bottleneck <https://pypi.python.org/pypi/Bottleneck>`_ package.
        kwargs : dict
            Passed to the numpy function.

        Returns
        -------
        result : :class:`~spectral_cube.lower_dimensional_structures.Projection` or `~astropy.units.Quantity` or float
            The result depends on the value of ``axis``, ``projection``, and
            ``unit``.  If ``axis`` is None, the return will be a scalar with or
            without units.  If axis is an integer, the return will be a
            :class:`~spectral_cube.lower_dimensional_structures.Projection` if ``projection`` is set
        """


        # leave axis in kwargs to avoid overriding numpy defaults, e.g.  if the
        # default is axis=-1, we don't want to force it to be axis=None by
        # specifying that in the function definition
        axis = kwargs.get('axis', None)

        log.debug("applying dask function {0}".format(function))

        import dask.array
        import dask.distributed
        try:
            client = dask.distributed.get_client()
        except ValueError:
            client = dask.distributed.Client()

        data_iterator = [dask.delayed(self._get_filled_data)(view=(ii,
                                                                   slice(None),
                                                                   slice(None)))
                         for ii in range(len(self))]
        list_of_futures = [dask.delayed(function)(x, **kwargs) for x in data_iterator]

        # Are there any cases where the output dtype != self.dtype?  FFTs, for
        # example, definitely fit this...
        list_of_arrays = [dask.array.from_delayed(fut, shape=self.shape[1:],
                                                  dtype=self.dtype) for fut in
                          list_of_futures]
        # we need a dask-array-like object so we can use `store`, since dask
        # doesn't (afaict) provide _any_ other ways to write to disk
        daskarr = dask.array.stack(list_of_arrays)

        if use_memmap:
            # create a memory map object to hold the data
            ntf = tempfile.NamedTemporaryFile(dir=memmap_dir)
            out = np.memmap(filename=ntf.name, mode='w+', shape=self.shape,
                            dtype=self.dtype)
            # clear the memmap object now that it has shape so we can "safely"
            # (hmm...) load it in other processes
            del out

            # This seems like the biggest hack ever
            out = DelayedMemmapWriter(ntf.name, self.shape, self.dtype)

            # instead of loading the data into memory, write them to our "out"
            # object, which is a memmap (wrapped by a DelayedMemmapWriter...)
            result = dask.array.store(daskarr,
                                      out,
                                      lock=False,
                                      compute=True)

            # important: re-load "out" from disk here, since "out" was defined
            # just a moment ago to be a non-array-like object
            out = np.memmap(filename=ntf.name, mode='r+', shape=self.shape,
                            dtype=self.dtype)
        else:
            # the easy way: if results fit in memory, just let dask compute them
            # and hand them back
            out = daskarr.compute()

        return self._reformat_cube_output(out, axis, unit,
                                          projection=projection, reduce=reduce)



    def _reduce_slicewise(self, function, fill, check_endian,
                          includemask=False, progressbar=False, **kwargs):
        """
        Compute a numpy aggregation by grabbing one slice at a time
        """

        ax = kwargs.pop('axis', None)
        full_reduce = ax is None
        ax = ax or 0

        if isinstance(ax, tuple):
            assert len(ax) == 2 # we only work with cubes...
            iterax = [x for x in range(3) if x not in ax][0]
        else:
            iterax = ax

        log.debug("reducing slicewise with axis = {0}".format(ax))

        if includemask:
            planes = self._iter_mask_slices(iterax)
        else:
            planes = self._iter_slices(iterax, fill=fill, check_endian=check_endian)
        result = next(planes)

        if progressbar:
            progressbar = ProgressBar(self.shape[iterax])
            pbu = progressbar.update
        else:
            pbu = lambda: True

        if isinstance(ax, tuple):
            # have to make a result a list of itself, since we already "got"
            # the first plane above
            result = [function(result, axis=(0,1), **kwargs)]
            for plane in planes:
                # apply to axes 0 and 1, because we're fully reducing the plane
                # to a number if we're applying over two axes
                result.append(function(plane, axis=(0,1), **kwargs))
                pbu()
            result = np.array(result)
        else:
            for plane in planes:
                # axis = 2 means we're stacking two planes, the previously
                # computed one and the current one
                result = function(np.dstack((result, plane)), axis=2, **kwargs)
                pbu()

        if full_reduce:
            result = function(result)

        return result

    def get_mask_array(self):
        """
        Convert the mask to a boolean numpy array
        """
        return self._mask.include(data=self._data, wcs=self._wcs,
                                  wcs_tolerance=self._wcs_tolerance)

    def _naxes_dropped(self, view):
        """
        Determine how many axes are being selected given a view.

        (1,2) -> 2
        None -> 3
        1 -> 1
        2 -> 1
        """

        if hasattr(view,'__len__'):
            return len(view)
        elif view is None:
            return 3
        else:
            return 1

    @aggregation_docstring
    @warn_slow
    def sum(self, axis=None, how='auto', **kwargs):
        """
        Return the sum of the cube, optionally over an axis.
        """
        from .np_compat import allbadtonan

        projection = self._naxes_dropped(axis) in (1,2)

        return self.apply_numpy_function(allbadtonan(np.nansum), fill=np.nan,
                                         how=how, axis=axis, unit=self.unit,
                                         projection=projection, **kwargs)

    @aggregation_docstring
    @warn_slow
    def mean(self, axis=None, how='cube', **kwargs):
        """
        Return the mean of the cube, optionally over an axis.
        """

        projection = self._naxes_dropped(axis) in (1,2)

        if how == 'slice':
            # two-pass approach: first total the # of points,
            # then total the value of the points, then divide
            # (a one-pass approach is possible but requires
            # more sophisticated bookkeeping)
            counts = self._count_nonzero_slicewise(axis=axis,
                                                   progressbar=kwargs.get('progressbar'))
            ttl = self.apply_numpy_function(np.nansum, fill=np.nan, how=how,
                                            axis=axis, unit=None,
                                            projection=False, **kwargs)
            out = ttl / counts
            if projection:
                if self._naxes_dropped(axis) == 1:
                    new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])
                    meta = {'collapse_axis': axis}
                    meta.update(self._meta)
                    return Projection(out, copy=False, wcs=new_wcs,
                                      meta=meta,
                                      unit=self.unit, header=self._nowcs_header)
                elif axis == (1,2):
                    newwcs = self._wcs.sub([wcs.WCSSUB_SPECTRAL])
                    if hasattr(self, '_beam') and self._beam is not None:
                        bmarg = {'beam': self.beam}
                    elif hasattr(self, 'beams'):
                        bmarg = {'beams': self.unmasked_beams}
                    else:
                        bmarg = {}
                    return self._oned_spectrum(value=out,
                                               wcs=newwcs,
                                               copy=False,
                                               unit=self.unit,
                                               spectral_unit=self._spectral_unit,
                                               meta=self.meta,
                                               **bmarg
                                              )
                else:
                    # this is a weird case, but even if projection is
                    # specified, we can't return a Quantity here because of WCS
                    # issues.  `apply_numpy_function` already does this
                    # silently, which is unfortunate.
                    warnings.warn("Averaging over a spatial and a spectral "
                                  "dimension cannot produce a Projection "
                                  "quantity (no units or WCS are preserved).",
                                  SliceWarning
                                 )
                    return out
            else:
                return out

        return self.apply_numpy_function(np.nanmean, fill=np.nan, how=how,
                                         axis=axis, unit=self.unit,
                                         projection=projection, **kwargs)

    def _count_nonzero_slicewise(self, axis=None, progressbar=False):
        """
        Count the number of finite pixels along an axis slicewise.  This is a
        helper function for the mean and std deviation slicewise iterators.
        """
        counts = self.apply_numpy_function(np.sum, fill=np.nan,
                                           how='slice', axis=axis,
                                           unit=None,
                                           projection=False,
                                           progressbar=progressbar,
                                           includemask=True)
        return counts

    @aggregation_docstring
    @warn_slow
    def std(self, axis=None, how='cube', ddof=0, **kwargs):
        """
        Return the standard deviation of the cube, optionally over an axis.

        Other Parameters
        ----------------
        ddof : int
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.  By
            default ``ddof`` is zero.
        """

        projection = self._naxes_dropped(axis) in (1,2)

        if how == 'slice':
            if axis is None:
                raise NotImplementedError("The overall standard deviation "
                                          "cannot be computed in a slicewise "
                                          "manner.  Please use a "
                                          "different strategy.")
            if hasattr(axis, '__len__') and len(axis) == 2:
                return self.apply_numpy_function(np.nanstd,
                                                 axis=axis,
                                                 how='slice',
                                                 projection=projection,
                                                 unit=self.unit,
                                                 **kwargs)
            else:
                counts = self._count_nonzero_slicewise(axis=axis)
                ttl = self.apply_numpy_function(np.nansum, fill=np.nan, how='slice',
                                                axis=axis, unit=None,
                                                projection=False, **kwargs)
                # Equivalent, but with more overhead:
                # ttl = self.sum(axis=axis, how='slice').value
                mean = ttl/counts

                planes = self._iter_slices(axis, fill=np.nan, check_endian=False)
                result = (next(planes)-mean)**2
                for plane in planes:
                    result = np.nansum(np.dstack((result, (plane-mean)**2)), axis=2)

                out = (result/(counts-ddof))**0.5

                if projection:
                    new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])
                    meta = {'collapse_axis': axis}
                    meta.update(self._meta)
                    return Projection(out, copy=False, wcs=new_wcs,
                                      meta=meta,
                                      unit=self.unit, header=self._nowcs_header)
                else:
                    return out

        # standard deviation cannot be computed as a trivial step-by-step
        # process.  There IS a one-pass algorithm for std dev, but it is not
        # implemented, so we must force cube here.  We could and should also
        # implement raywise reduction
        return self.apply_numpy_function(np.nanstd, fill=np.nan, how=how,
                                         axis=axis, unit=self.unit,
                                         projection=projection, **kwargs)

    @aggregation_docstring
    @warn_slow
    def mad_std(self, axis=None, how='cube', **kwargs):
        """
        Use astropy's mad_std to computer the standard deviation
        """
        if int(astropy.__version__[0]) < 2:
            raise NotImplementedError("mad_std requires astropy >= 2")
        projection = self._naxes_dropped(axis) in (1,2)
        if how == 'ray' and not hasattr(axis, '__len__'):
            # no need for fill here; masked-out data are simply not included
            return self.apply_numpy_function(stats.mad_std,
                                             axis=axis,
                                             how='ray',
                                             unit=self.unit,
                                             projection=projection,
                                             ignore_nan=True,
                                            )
        elif how == 'slice' and hasattr(axis, '__len__') and len(axis) == 2:
            return self.apply_numpy_function(stats.mad_std,
                                             axis=axis,
                                             how='slice',
                                             projection=projection,
                                             unit=self.unit,
                                             fill=np.nan,
                                             ignore_nan=True,
                                             **kwargs)
        elif how in ('ray', 'slice'):
            raise NotImplementedError('Cannot run mad_std slicewise or raywise '
                                      'unless the dimensionality is also reduced in the same direction.')
        else:
            return self.apply_numpy_function(stats.mad_std,
                                             fill=np.nan,
                                             axis=axis,
                                             unit=self.unit,
                                             ignore_nan=True,
                                             how=how,
                                             projection=projection, **kwargs)


    @aggregation_docstring
    @warn_slow
    def max(self, axis=None, how='auto', **kwargs):
        """
        Return the maximum data value of the cube, optionally over an axis.
        """

        projection = self._naxes_dropped(axis) in (1,2)

        return self.apply_numpy_function(np.nanmax, fill=np.nan, how=how,
                                         axis=axis, unit=self.unit,
                                         projection=projection, **kwargs)

    @aggregation_docstring
    @warn_slow
    def min(self, axis=None, how='auto', **kwargs):
        """
        Return the minimum data value of the cube, optionally over an axis.
        """

        projection = self._naxes_dropped(axis) in (1,2)

        return self.apply_numpy_function(np.nanmin, fill=np.nan, how=how,
                                         axis=axis, unit=self.unit,
                                         projection=projection, **kwargs)

    @aggregation_docstring
    @warn_slow
    def argmax(self, axis=None, how='auto', **kwargs):
        """
        Return the index of the maximum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return self.apply_numpy_function(np.nanargmax, fill=-np.inf,
                                         reduce=False, projection=False,
                                         how=how, axis=axis, **kwargs)

    @aggregation_docstring
    @warn_slow
    def argmin(self, axis=None, how='auto', **kwargs):
        """
        Return the index of the minimum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask
        """
        return self.apply_numpy_function(np.nanargmin, fill=np.inf,
                                         reduce=False, projection=False,
                                         how=how, axis=axis, **kwargs)

    def chunked(self, chunksize=1000):
        """
        Not Implemented.

        Iterate over chunks of valid data
        """
        raise NotImplementedError()

    def _get_flat_shape(self, axis):
        """
        Get the shape of the array after flattening along an axis
        """
        iteraxes = [0, 1, 2]
        iteraxes.remove(axis)
        # x,y are defined as first,second dim to iterate over
        # (not x,y in pixel space...)
        nx = self.shape[iteraxes[0]]
        ny = self.shape[iteraxes[1]]
        return nx, ny

    @warn_slow
    def _apply_everywhere(self, function, *args):
        """
        Return a new cube with ``function`` applied to all pixels

        Private because this doesn't have an obvious and easy-to-use API

        Examples
        --------
        >>> newcube = cube.apply_everywhere(np.add, 0.5*u.Jy)
        """

        try:
            test_result = function(np.ones([1,1,1])*self.unit, *args)
            # First, check that function returns same # of dims?
            assert test_result.ndim == 3,"Output is not 3-dimensional"
        except Exception as ex:
            raise AssertionError("Function could not be applied to a simple "
                                 "cube.  The error was: {0}".format(ex))

        data = function(u.Quantity(self._get_filled_data(fill=self._fill_value),
                                   self.unit, copy=False),
                        *args)

        return self._new_cube_with(data=data, unit=data.unit)

    @warn_slow
    def _cube_on_cube_operation(self, function, cube, equivalencies=[], **kwargs):
        """
        Apply an operation between two cubes.  Inherits the metadata of the
        left cube.

        Parameters
        ----------
        function : function
            A function to apply to the cubes
        cube : SpectralCube
            Another cube to put into the function
        equivalencies : list
            A list of astropy equivalencies
        kwargs : dict
            Passed to np.testing.assert_almost_equal
        """
        assert cube.shape == self.shape
        if not self.unit.is_equivalent(cube.unit, equivalencies=equivalencies):
            raise u.UnitsError("{0} is not equivalent to {1}"
                               .format(self.unit, cube.unit))
        if not wcs_utils.check_equality(self.wcs, cube.wcs, warn_missing=True,
                                        **kwargs):
            warnings.warn("Cube WCSs do not match, but their shapes do",
                          WCSMismatchWarning)
        try:
            test_result = function(np.ones([1,1,1])*self.unit,
                                   np.ones([1,1,1])*self.unit)
            # First, check that function returns same # of dims?
            assert test_result.shape == (1,1,1)
        except Exception as ex:
            raise AssertionError("Function {1} could not be applied to a "
                                 "pair of simple "
                                 "cube.  The error was: {0}".format(ex,
                                                                    function))

        cube = cube.to(self.unit)
        data = function(self._data, cube._data)
        try:
            # multiplication, division, etc. are valid inter-unit operations
            unit = function(self.unit, cube.unit)
        except TypeError:
            # addition, subtraction are not
            unit = self.unit

        return self._new_cube_with(data=data, unit=unit)

    def apply_function(self, function, axis=None, weights=None, unit=None,
                       projection=False, progressbar=False,
                       update_function=None, keep_shape=False, **kwargs):
        """
        Apply a function to valid data along the specified axis or to the whole
        cube, optionally using a weight array that is the same shape (or at
        least can be sliced in the same way)

        Parameters
        ----------
        function : function
            A function that can be applied to a numpy array.  Does not need to
            be nan-aware
        axis : 1, 2, 3, or None
            The axis to operate along.  If None, the return is scalar.
        weights : (optional) np.ndarray
            An array with the same shape (or slicing abilities/results) as the
            data cube
        unit : (optional) `~astropy.units.Unit`
            The unit of the output projection or value.  Not all functions
            should return quantities with units.
        projection : bool
            Return a projection if the resulting array is 2D?
        progressbar : bool
            Show a progressbar while iterating over the slices/rays through the
            cube?
        keep_shape : bool
            If `True`, the returned object will be the same dimensionality as
            the cube.
        update_function : function
            An alternative tracker for the progress of applying the function
            to the cube data. If ``progressbar`` is ``True``, this argument is
            ignored.

        Returns
        -------
        result : :class:`~spectral_cube.lower_dimensional_structures.Projection` or `~astropy.units.Quantity` or float
            The result depends on the value of ``axis``, ``projection``, and
            ``unit``.  If ``axis`` is None, the return will be a scalar with or
            without units.  If axis is an integer, the return will be a
            :class:`~spectral_cube.lower_dimensional_structures.Projection` if ``projection`` is set
        """
        if axis is None:
            out = function(self.flattened(), **kwargs)
            if unit is not None:
                return u.Quantity(out, unit=unit)
            else:
                return out
        if hasattr(axis, '__len__'):
            raise NotImplementedError("`apply_function` does not support "
                                      "function application across multiple "
                                      "axes.  Try `apply_numpy_function`.")

        # determine the output array shape
        nx, ny = self._get_flat_shape(axis)
        nz = self.shape[axis] if keep_shape else 1

        # allocate memory for output array
        out = np.empty([nz, nx, ny]) * np.nan

        if progressbar:
            progressbar = ProgressBar(nx*ny)
            pbu = progressbar.update
        elif update_function is not None:
            pbu = update_function
        else:
            pbu = lambda: True

        # iterate over "lines of sight" through the cube
        for y, x, slc in self._iter_rays(axis):
            # acquire the flattened, valid data for the slice
            data = self.flattened(slc, weights=weights)
            if len(data) != 0:
                result = function(data, **kwargs)
                if hasattr(result, 'value'):
                    # store result in array
                    out[:, y, x] = result.value
                else:
                    out[:, y, x] = result
            pbu()

        if not keep_shape:
            out = out[0, :, :]

        if projection and axis in (0, 1, 2):
            new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])

            meta = {'collapse_axis': axis}
            meta.update(self._meta)

            return Projection(out, copy=False, wcs=new_wcs, meta=meta,
                              unit=unit, header=self._nowcs_header)
        else:
            return out

    def _iter_rays(self, axis=None):
        """
        Iterate over view corresponding to lines-of-sight through a cube
        along the specified axis
        """
        ny, nx = self._get_flat_shape(axis)

        for y in range(ny):
            for x in range(nx):
                # create length-1 view for each position
                slc = [slice(y, y + 1), slice(x, x + 1), ]
                # create a length-N slice (all-inclusive) along the selected axis
                slc.insert(axis, slice(None))
                yield y, x, tuple(slc)

    def _iter_slices(self, axis, fill=np.nan, check_endian=False):
        """
        Iterate over the cube one slice at a time,
        replacing masked elements with fill
        """
        view = [slice(None)] * 3
        for x in range(self.shape[axis]):
            view[axis] = x
            yield self._get_filled_data(view=tuple(view), fill=fill,
                                        check_endian=check_endian)

    def _iter_mask_slices(self, axis):
        """
        Iterate over the cube one slice at a time,
        replacing masked elements with fill
        """
        view = [slice(None)] * 3
        for x in range(self.shape[axis]):
            view[axis] = x
            yield self._mask.include(data=self._data,
                                     view=tuple(view),
                                     wcs=self._wcs,
                                     wcs_tolerance=self._wcs_tolerance,
                                    )

    def flattened(self, slice=(), weights=None):
        """
        Return a slice of the cube giving only the valid data (i.e., removing
        bad values)

        Parameters
        ----------
        slice: 3-tuple
            A length-3 tuple of view (or any equivalent valid slice of a
            cube)
        weights: (optional) np.ndarray
            An array with the same shape (or slicing abilities/results) as the
            data cube
        """
        data = self._mask._flattened(data=self._data, wcs=self._wcs, view=slice)
        if weights is not None:
            weights = self._mask._flattened(data=weights, wcs=self._wcs, view=slice)
            return u.Quantity(data * weights, self.unit, copy=False)
        else:
            return u.Quantity(data, self.unit, copy=False)

    def median(self, axis=None, iterate_rays=False, **kwargs):
        """
        Compute the median of an array, optionally along an axis.

        Ignores excluded mask elements.

        Parameters
        ----------
        axis : int (optional)
            The axis to collapse
        iterate_rays : bool
            Iterate over individual rays?  This mode is slower but can save RAM
            costs, which may be extreme for large cubes

        Returns
        -------
        med : ndarray
            The median
        """
        try:
            from bottleneck import nanmedian
            bnok = True
        except ImportError:
            bnok = False

        # slicewise median is nonsense, must force how = 'cube'
        # bottleneck.nanmedian does not allow axis to be a list or tuple
        if bnok and not iterate_rays and not isinstance(axis, (list, tuple)):
            log.debug("Using bottleneck nanmedian")
            result = self.apply_numpy_function(nanmedian, axis=axis,
                                               projection=True, unit=self.unit,
                                               how='cube', check_endian=True,
                                               **kwargs)
        elif hasattr(np, 'nanmedian') and not iterate_rays:
            log.debug("Using numpy nanmedian")
            result = self.apply_numpy_function(np.nanmedian, axis=axis,
                                               projection=True, unit=self.unit,
                                               how='cube',**kwargs)
        else:
            log.debug("Using numpy median iterating over rays")
            result = self.apply_function(np.median, projection=True, axis=axis,
                                         unit=self.unit, **kwargs)

        return result

    def percentile(self, q, axis=None, iterate_rays=False, **kwargs):
        """
        Return percentiles of the data.

        Parameters
        ----------
        q : float
            The percentile to compute
        axis : int, or None
            Which axis to compute percentiles over
        iterate_rays : bool
            Iterate over individual rays?  This mode is slower but can save RAM
            costs, which may be extreme for large cubes
        """
        if hasattr(np, 'nanpercentile') and not iterate_rays:
            result = self.apply_numpy_function(np.nanpercentile, q=q,
                                               axis=axis, projection=True,
                                               unit=self.unit, how='cube',
                                               **kwargs)
        else:
            result = self.apply_function(np.percentile, q=q, axis=axis,
                                         projection=True, unit=self.unit,
                                         **kwargs)

        return result

    def with_mask(self, mask, inherit_mask=True, wcs_tolerance=None):
        """
        Return a new SpectralCube instance that contains a composite mask of
        the current SpectralCube and the new ``mask``.  Values of the mask that
        are ``True`` will be *included* (masks are analogous to numpy boolean
        index arrays, they are the inverse of the ``.mask`` attribute of a numpy
        masked array).

        Parameters
        ----------
        mask : :class:`~spectral_cube.masks.MaskBase` instance, or boolean numpy array
            The mask to apply. If a boolean array is supplied,
            it will be converted into a mask, assuming that
            `True` values indicate included elements.

        inherit_mask : bool (optional, default=True)
            If True, combines the provided mask with the
            mask currently attached to the cube

        wcs_tolerance : None or float
            The tolerance of difference in WCS parameters between the cube and
            the mask.  Defaults to `self._wcs_tolerance` (which itself defaults
            to 0.0) if unspecified

        Returns
        -------
        new_cube : :class:`SpectralCube`
            A cube with the new mask applied.

        Notes
        -----
        This operation returns a view into the data, and not a copy.
        """
        if isinstance(mask, np.ndarray):
            if not is_broadcastable_and_smaller(mask.shape, self._data.shape):
                raise ValueError("Mask shape is not broadcastable to data shape: "
                                 "%s vs %s" % (mask.shape, self._data.shape))
            mask = BooleanArrayMask(mask, self._wcs, shape=self._data.shape)

        if self._mask is not None and inherit_mask:
            new_mask = np.bitwise_and(self._mask, mask)
        else:
            new_mask = mask

        new_mask._validate_wcs(new_data=self._data, new_wcs=self._wcs,
                               wcs_tolerance=wcs_tolerance or self._wcs_tolerance)

        return self._new_cube_with(mask=new_mask, wcs_tolerance=wcs_tolerance)

    def __getitem__(self, view):

        # Need to allow self[:], self[:,:]
        if isinstance(view, (slice,int,np.int64)):
            view = (view, slice(None), slice(None))
        elif len(view) == 2:
            view = view + (slice(None),)
        elif len(view) > 3:
            raise IndexError("Too many indices")

        meta = {}
        meta.update(self._meta)
        slice_data = [(s.start, s.stop, s.step)
                      if hasattr(s,'start') else s
                      for s in view]
        if 'slice' in meta:
            meta['slice'].append(slice_data)
        else:
            meta['slice'] = [slice_data]

        intslices = [2-ii for ii,s in enumerate(view) if not hasattr(s,'start')]

        if intslices:
            if len(intslices) > 1:
                if 2 in intslices:
                    raise NotImplementedError("1D slices along non-spectral "
                                              "axes are not yet implemented.")
                newwcs = self._wcs.sub([a
                                        for a in (1,2,3)
                                        if a not in [x+1 for x in intslices]])
                # As of #561, beam is defined in all cases
                if hasattr(self, '_beam') and self._beam is not None:
                    bmarg = {'beam': self.beam}
                elif hasattr(self, 'beams'):
                    bmarg = {'beams': self.beams}
                else:
                    bmarg = {}
                return self._oned_spectrum(value=self._data[view],
                                           wcs=newwcs,
                                           copy=False,
                                           unit=self.unit,
                                           spectral_unit=self._spectral_unit,
                                           mask=self.mask[view],
                                           meta=meta,
                                           **bmarg
                                          )

            # only one element, so drop an axis
            newwcs = wcs_utils.drop_axis(self._wcs, intslices[0])
            header = self._nowcs_header

            if intslices[0] == 0:
                # celestial: can report the wavelength/frequency of the axis
                header['CRVAL3'] = self.spectral_axis[intslices[0]].value
                header['CDELT3'] = self.wcs.sub([wcs.WCSSUB_SPECTRAL]).wcs.cdelt[0]
                header['CUNIT3'] = self._spectral_unit.to_string(format='FITS')

            return Slice(value=self.filled_data[view],
                         mask=self.mask[view],
                         wcs=newwcs,
                         copy=False,
                         unit=self.unit,
                         header=header,
                         meta=meta)

        newmask = self._mask[view] if self._mask is not None else None

        newwcs = wcs_utils.slice_wcs(self._wcs, view, shape=self.shape)

        return self._new_cube_with(data=self._data[view],
                                   wcs=newwcs,
                                   mask=newmask,
                                   meta=meta)

    @property
    def unitless(self):
        """Return a copy of self with unit set to None"""
        newcube = self._new_cube_with()
        newcube._unit = None
        return newcube

    def with_spectral_unit(self, unit, velocity_convention=None,
                           rest_value=None):
        """
        Returns a new Cube with a different Spectral Axis unit

        Parameters
        ----------
        unit : :class:`~astropy.units.Unit`
            Any valid spectral unit: velocity, (wave)length, or frequency.
            Only vacuum units are supported.
        velocity_convention : 'relativistic', 'radio', or 'optical'
            The velocity convention to use for the output velocity axis.
            Required if the output type is velocity. This can be either one
            of the above strings, or an `astropy.units` equivalency.
        rest_value : :class:`~astropy.units.Quantity`
            A rest wavelength or frequency with appropriate units.  Required if
            output type is velocity.  The cube's WCS should include this
            already if the *input* type is velocity, but the WCS's rest
            wavelength/frequency can be overridden with this parameter.

            .. note: This must be the rest frequency/wavelength *in vacuum*,
                     even if your cube has air wavelength units

        """
        newwcs,newmeta = self._new_spectral_wcs(unit=unit,
                                                velocity_convention=velocity_convention,
                                                rest_value=rest_value)

        if self._mask is not None:
            newmask = self._mask.with_spectral_unit(unit,
                                                    velocity_convention=velocity_convention,
                                                    rest_value=rest_value)
            newmask._wcs = newwcs
        else:
            newmask = None


        cube = self._new_cube_with(wcs=newwcs, mask=newmask, meta=newmeta,
                                   spectral_unit=unit)

        return cube


    @cube_utils.slice_syntax
    def unmasked_data(self, view):
        """
        Return a view of the subset of the underlying data,
        ignoring the mask.

        Returns
        -------
        data : Quantity instance
            The unmasked data
        """
        values = self._data[view]
        # Astropy Quantities don't play well with dask arrays with shape ()
        if isinstance(values, dask.array.core.Array) and values.shape == ():
            values = values.compute()
        return u.Quantity(values, self.unit, copy=False)

    def unmasked_copy(self):
        """
        Return a copy of the cube with no mask (i.e., all data included)
        """
        newcube = self._new_cube_with()
        newcube._mask = None
        return newcube

    @cached
    def _pix_cen(self):
        """
        Offset of every pixel from the origin, along each direction

        Returns
        -------
        tuple of spectral_offset, y_offset, x_offset, each 3D arrays
        describing the distance from the origin

        Notes
        -----
        These arrays are broadcast, and are not memory intensive

        Each array is in the units of the corresponding wcs.cunit, but
        this is implicit (e.g., they are not astropy Quantity arrays)
        """
        # Start off by extracting the world coordinates of the pixels
        _, lat, lon = self.world[0, :, :]
        spectral, _, _ = self.world[:, 0, 0]
        spectral -= spectral[0] # offset from first pixel

        # Convert to radians
        lon = np.radians(lon)
        lat = np.radians(lat)

        # Find the dx and dy arrays
        from astropy.coordinates.angle_utilities import angular_separation
        dx = angular_separation(lon[:, :-1], lat[:, :-1],
                                lon[:, 1:], lat[:, :-1])
        dy = angular_separation(lon[:-1, :], lat[:-1, :],
                                lon[1:, :], lat[1:, :])

        # Find the cumulative offset - need to add a zero at the start
        x = np.zeros(self._data.shape[1:])
        y = np.zeros(self._data.shape[1:])
        x[:, 1:] = np.cumsum(np.degrees(dx), axis=1)
        y[1:, :] = np.cumsum(np.degrees(dy), axis=0)

        x, y, spectral = np.broadcast_arrays(x[None,:,:], y[None,:,:], spectral[:,None,None])

        return spectral, y, x

    @cached
    def _pix_size_slice(self, axis):
        """
        Return the size of each pixel along any given direction.  Assumes
        pixels have equal size.  Also assumes that the spectral and spatial
        directions are separable, which is enforced throughout this code.

        Parameters
        ----------
        axis : 0, 1, or 2
            The axis along which to compute the pixel size

        Returns
        -------
        Pixel size in units of either degrees or the appropriate spectral unit
        """
        if axis == 0:
            # note that self._spectral_scale is required here because wcs
            # forces into units of m, m/s, or Hz
            return np.abs(self.wcs.pixel_scale_matrix[2,2]) * self._spectral_scale
        elif axis in (1,2):
            # the pixel size is a projection.  I think the pixel_scale_matrix
            # must be symmetric, such that psm[axis,:]**2 == psm[:,axis]**2
            return np.sum(self.wcs.pixel_scale_matrix[2-axis,:]**2)**0.5
        else:
            raise ValueError("Cubes have 3 axes.")

    @cached
    def _pix_size(self):
        """
        Return the size of each pixel along each direction, in world units

        Returns
        -------
        dv, dy, dx : tuple of 3D arrays

        The extent of each pixel along each direction

        Notes
        -----
        These arrays are broadcast, and are not memory intensive

        Each array is in the units of the corresponding wcs.cunit, but
        this is implicit (e.g., they are not astropy Quantity arrays)
        """

        # First, scale along x direction

        xpix = np.linspace(-0.5, self._data.shape[2] - 0.5, self._data.shape[2] + 1)
        ypix = np.linspace(0., self._data.shape[1] - 1, self._data.shape[1])
        xpix, ypix = np.meshgrid(xpix, ypix)
        zpix = np.zeros(xpix.shape)

        lon, lat, _ = self._wcs.all_pix2world(xpix, ypix, zpix, 0)

        # Convert to radians
        lon = np.radians(lon)
        lat = np.radians(lat)

        # Find the dx and dy arrays
        from astropy.coordinates.angle_utilities import angular_separation
        dx = angular_separation(lon[:, :-1], lat[:, :-1],
                                lon[:, 1:], lat[:, :-1])

        # Next, scale along y direction

        xpix = np.linspace(0., self._data.shape[2] - 1, self._data.shape[2])
        ypix = np.linspace(-0.5,
                           self._data.shape[1] - 0.5,
                           self._data.shape[1] + 1)
        xpix, ypix = np.meshgrid(xpix, ypix)
        zpix = np.zeros(xpix.shape)

        lon, lat, _ = self._wcs.all_pix2world(xpix, ypix, zpix, 0)

        # Convert to radians
        lon = np.radians(lon)
        lat = np.radians(lat)

        # Find the dx and dy arrays
        from astropy.coordinates.angle_utilities import angular_separation
        dy = angular_separation(lon[:-1, :], lat[:-1, :],
                                lon[1:, :], lat[1:, :])

        # Next, spectral coordinates
        zpix = np.linspace(-0.5, self._data.shape[0] - 0.5,
                           self._data.shape[0] + 1)
        xpix = np.zeros(zpix.shape)
        ypix = np.zeros(zpix.shape)

        _, _, spectral = self._wcs.all_pix2world(xpix, ypix, zpix, 0)

        # Take spectral units into account
        # order of operations here is crucial!  If this is done after
        # broadcasting, the full array size is allocated, which is bad!
        dspectral = np.diff(spectral) * self._spectral_scale

        dx = np.abs(np.degrees(dx.reshape(1, dx.shape[0], dx.shape[1])))
        dy = np.abs(np.degrees(dy.reshape(1, dy.shape[0], dy.shape[1])))
        dspectral = np.abs(dspectral.reshape(-1, 1, 1))
        dx, dy, dspectral = np.broadcast_arrays(dx, dy, dspectral)

        return dspectral, dy, dx

    def moment(self, order=0, axis=0, how='auto'):
        """
        Compute moments along the spectral axis.

        Moments are defined as follows:

        Moment 0:

        .. math:: M_0 \\int I dl

        Moment 1:

        .. math:: M_1 = \\frac{\\int I l dl}{M_0}

        Moment N:

        .. math:: M_N = \\frac{\\int I (l - M_1)^N dl}{M_0}

        .. warning:: Note that these follow the mathematical definitions of
                     moments, and therefore the second moment will return a
                     variance map. To get linewidth maps, you can instead use
                     the :meth:`~SpectralCube.linewidth_fwhm` or
                     :meth:`~SpectralCube.linewidth_sigma` methods.

        Parameters
        ----------
        order : int
           The order of the moment to take. Default=0

        axis : int
           The axis along which to compute the moment. Default=0

        how : cube | slice | ray | auto
           How to compute the moment. All strategies give the same
           result, but certain strategies are more efficient depending
           on data size and layout. Cube/slice/ray iterate over
           decreasing subsets of the data, to conserve memory.
           Default='auto'

        Returns
        -------
           map [, wcs]
           The moment map (numpy array) and, if wcs=True, the WCS object
           describing the map

        Notes
        -----
        Generally, how='cube' is fastest for small cubes that easily
        fit into memory. how='slice' is best for most larger datasets.
        how='ray' is probably only a good idea for very large cubes
        whose data are contiguous over the axis of the moment map.

        For the first moment, the result for axis=1, 2 is the angular
        offset *relative to the cube face*. For axis=0, it is the
        *absolute* velocity/frequency of the first moment.
        """

        if axis == 0 and order == 2:
            warnings.warn("Note that the second moment returned will be a "
                          "variance map. To get a linewidth map, use the "
                          "SpectralCube.linewidth_fwhm() or "
                          "SpectralCube.linewidth_sigma() methods instead.",
                          VarianceWarning)

        from ._moments import (moment_slicewise, moment_cubewise,
                               moment_raywise, moment_auto)

        dispatch = dict(slice=moment_slicewise,
                        cube=moment_cubewise,
                        ray=moment_raywise,
                        auto=moment_auto)

        if how not in dispatch:
            return ValueError("Invalid how. Must be in %s" %
                              sorted(list(dispatch.keys())))

        out = dispatch[how](self, order, axis)

        # apply units
        if order == 0:
            if axis == 0 and self._spectral_unit is not None:
                axunit = unit = self._spectral_unit
            else:
                axunit = unit = u.Unit(self._wcs.wcs.cunit[np2wcs[axis]])
            out = u.Quantity(out, self.unit * axunit, copy=False)
        else:
            if axis == 0 and self._spectral_unit is not None:
                unit = self._spectral_unit ** max(order, 1)
            else:
                unit = u.Unit(self._wcs.wcs.cunit[np2wcs[axis]]) ** max(order, 1)
            out = u.Quantity(out, unit, copy=False)

        # special case: for order=1, axis=0, you usually want
        # the absolute velocity and not the offset
        if order == 1 and axis == 0:
            out += self.world[0, :, :][0]

        new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])

        meta = {'moment_order': order,
                'moment_axis': axis,
                'moment_method': how}
        meta.update(self._meta)

        return Projection(out, copy=False, wcs=new_wcs, meta=meta,
                          header=self._nowcs_header)

    def moment0(self, axis=0, how='auto'):
        """
        Compute the zeroth moment along an axis.

        See :meth:`moment`.
        """
        return self.moment(axis=axis, order=0, how=how)

    def moment1(self, axis=0, how='auto'):
        """
        Compute the 1st moment along an axis.

        For an explanation of the ``axis`` and ``how`` parameters, see :meth:`moment`.
        """
        return self.moment(axis=axis, order=1, how=how)

    def moment2(self, axis=0, how='auto'):
        """
        Compute the 2nd moment along an axis.

        For an explanation of the ``axis`` and ``how`` parameters, see :meth:`moment`.
        """
        return self.moment(axis=axis, order=2, how=how)

    def linewidth_sigma(self, how='auto'):
        """
        Compute a (sigma) linewidth map along the spectral axis.

        For an explanation of the ``how`` parameter, see :meth:`moment`.
        """
        with np.errstate(invalid='ignore'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", VarianceWarning)
                return np.sqrt(self.moment2(how=how))

    def linewidth_fwhm(self, how='auto'):
        """
        Compute a (FWHM) linewidth map along the spectral axis.

        For an explanation of the ``how`` parameter, see :meth:`moment`.
        """
        return self.linewidth_sigma() * SIGMA2FWHM

    @property
    def spectral_axis(self):
        """
        A `~astropy.units.Quantity` array containing the central values of
        each channel along the spectral axis.
        """
        return self.world[:, 0, 0][0].ravel()

    @property
    def velocity_convention(self):
        """
        The `~astropy.units.equivalencies` that describes the spectral axis
        """
        return spectral_axis.determine_vconv_from_ctype(self.wcs.wcs.ctype[self.wcs.wcs.spec])

    def closest_spectral_channel(self, value):
        """
        Find the index of the closest spectral channel to the specified
        spectral coordinate.

        Parameters
        ----------
        value : :class:`~astropy.units.Quantity`
            The value of the spectral coordinate to search for.
        """

        # TODO: we have to not compute this every time
        spectral_axis = self.spectral_axis

        try:
            value = value.to(spectral_axis.unit, equivalencies=u.spectral())
        except u.UnitsError:
            if value.unit.is_equivalent(u.Hz, equivalencies=u.spectral()):
                if spectral_axis.unit.is_equivalent(u.m / u.s):
                    raise u.UnitsError("Spectral axis is in velocity units and "
                                       "'value' is in frequency-equivalent units "
                                       "- use SpectralCube.with_spectral_unit "
                                       "first to convert the cube to frequency-"
                                       "equivalent units, or search for a "
                                       "velocity instead")
                else:
                    raise u.UnitsError("Unexpected spectral axis units: {0}".format(spectral_axis.unit))
            elif value.unit.is_equivalent(u.m / u.s):
                if spectral_axis.unit.is_equivalent(u.Hz, equivalencies=u.spectral()):
                    raise u.UnitsError("Spectral axis is in frequency-equivalent "
                                       "units and 'value' is in velocity units "
                                       "- use SpectralCube.with_spectral_unit "
                                       "first to convert the cube to frequency-"
                                       "equivalent units, or search for a "
                                       "velocity instead")
                else:
                    raise u.UnitsError("Unexpected spectral axis units: {0}".format(spectral_axis.unit))
            else:
                raise u.UnitsError("'value' should be in frequency equivalent or velocity units (got {0})".format(value.unit))

        # TODO: optimize the next line - just brute force for now
        return np.argmin(np.abs(spectral_axis - value))

    def spectral_slab(self, lo, hi):
        """
        Extract a new cube between two spectral coordinates

        Parameters
        ----------
        lo, hi : :class:`~astropy.units.Quantity`
            The lower and upper spectral coordinate for the slab range. The
            units should be compatible with the units of the spectral axis.
            If the spectral axis is in frequency-equivalent units and you
            want to select a range in velocity, or vice-versa, you should
            first use :meth:`~spectral_cube.SpectralCube.with_spectral_unit`
            to convert the units of the spectral axis.
        """

        # Find range of values for spectral axis
        ilo = self.closest_spectral_channel(lo)
        ihi = self.closest_spectral_channel(hi)

        if ilo > ihi:
            ilo, ihi = ihi, ilo
        ihi += 1

        # Create WCS slab
        wcs_slab = self._wcs.deepcopy()
        wcs_slab.wcs.crpix[2] -= ilo

        # Create mask slab
        if self._mask is None:
            mask_slab = None
        else:
            try:
                mask_slab = self._mask[ilo:ihi, :, :]
            except NotImplementedError:
                warnings.warn("Mask slicing not implemented for "
                              "{0} - dropping mask".
                              format(self._mask.__class__.__name__),
                              NotImplementedWarning
                             )
                mask_slab = None

        # Create new spectral cube
        slab = self._new_cube_with(data=self._data[ilo:ihi], wcs=wcs_slab,
                                   mask=mask_slab)

        # TODO: we could change the WCS to give a spectral axis in the
        # correct units as requested - so if the initial cube is in Hz and we
        # request a range in km/s, we could adjust the WCS to be in km/s
        # instead

        return slab

    def minimal_subcube(self, spatial_only=False):
        """
        Return the minimum enclosing subcube where the mask is valid

        Parameters
        ----------
        spatial_only: bool
            Only compute the minimal subcube in the spatial dimensions
        """
        return self[self.subcube_slices_from_mask(self._mask,
                                                  spatial_only=spatial_only)]

    def subcube_from_mask(self, region_mask):
        """
        Given a mask, return the minimal subcube that encloses the mask

        Parameters
        ----------
        region_mask: `~spectral_cube.masks.MaskBase` or boolean `numpy.ndarray`
            The mask with appropraite WCS or an ndarray with matched
            coordinates
        """
        return self[self.subcube_slices_from_mask(region_mask)]

    def subcube_slices_from_mask(self, region_mask, spatial_only=False):
        """
        Given a mask, return the slices corresponding to the minimum subcube
        that encloses the mask

        Parameters
        ----------
        region_mask: `~spectral_cube.masks.MaskBase` or boolean `numpy.ndarray`
            The mask with appropriate WCS or an ndarray with matched
            coordinates
        spatial_only: bool
            Return only slices that affect the spatial dimensions; the spectral
            dimension will be left unchanged
        """
        if not scipyOK:
            raise ImportError("Scipy could not be imported: this function won't work.")

        if isinstance(region_mask, np.ndarray):
            if is_broadcastable_and_smaller(region_mask.shape, self.shape):
                region_mask = BooleanArrayMask(region_mask, self._wcs)
            else:
                raise ValueError("Mask shape does not match cube shape.")

        include = region_mask.include(self._data, self._wcs,
                                      wcs_tolerance=self._wcs_tolerance)

        if not include.any():
            return (slice(0),)*3

        slices = ndimage.find_objects(np.broadcast_arrays(include,
                                                          self._data)[0])[0]

        if spatial_only:
            slices = (slice(None), slices[1], slices[2])

        return tuple(slices)

    def subcube(self, xlo='min', xhi='max', ylo='min', yhi='max', zlo='min',
                zhi='max', rest_value=None):
        """
        Extract a sub-cube spatially and spectrally.

        Parameters
        ----------
        [xyz]lo/[xyz]hi : int or :class:`~astropy.units.Quantity` or ``min``/``max``
            The endpoints to extract.  If given as a quantity, will be
            interpreted as World coordinates.  If given as a string or
            int, will be interpreted as pixel coordinates.
        """

        limit_dict = {'xlo':0 if xlo == 'min' else xlo,
                      'ylo':0 if ylo == 'min' else ylo,
                      'zlo':0 if zlo == 'min' else zlo,
                      'xhi':self.shape[2] if xhi=='max' else xhi,
                      'yhi':self.shape[1] if yhi=='max' else yhi,
                      'zhi':self.shape[0] if zhi=='max' else zhi}
        dims = {'x': 2,
                'y': 1,
                'z': 0}

        # Specific warning for slicing a frequency axis with a velocity or
        # vice/versa
        if ((hasattr(zlo, 'unit') and not
             zlo.unit.is_equivalent(self.spectral_axis.unit)) or
            (hasattr(zhi, 'unit') and not
             zhi.unit.is_equivalent(self.spectral_axis.unit))):
            raise u.UnitsError("Spectral units are not equivalent to the "
                               "spectral slice.  Use `.with_spectral_unit` "
                               "to convert to equivalent units first")

        for val in (xlo,ylo,xhi,yhi):
            if hasattr(val, 'unit') and not val.unit.is_equivalent(u.degree):
                raise u.UnitsError("The X and Y slices must be specified in "
                                   "degree-equivalent units.")

        # list to track which entries had units
        united = []

        for lim in limit_dict:
            limval = limit_dict[lim]
            if hasattr(limval, 'unit'):
                united.append(lim)
                dim = dims[lim[0]]
                sl = [slice(0,1)]*2
                sl.insert(dim, slice(None))
                sl = tuple(sl)
                spine = self.world[sl][dim]
                val = np.argmin(np.abs(limval-spine))
                if limval > spine.max() or limval < spine.min():
                    log.warning("The limit {0} is out of bounds."
                                "  Using min/max instead.".format(lim))
                limit_dict[lim] = val

        for xx in 'zyx':
            hi,lo = limit_dict[xx+'hi'], limit_dict[xx+'lo']
            if hi < lo:
                # must have high > low
                limit_dict[xx+'hi'], limit_dict[xx+'lo'] = lo, hi

            if xx+'lo' in united:
                # End-inclusive indexing: need to add one for the high slice
                # Only do this for converted values, not for pixel values
                # (i.e., if the xlo/ylo/zlo value had units)
                limit_dict[xx+'hi'] += 1

        for xx in 'zyx':
            if limit_dict[xx+'hi'] == limit_dict[xx+'lo']:
                # I think this should be unreachable now
                raise ValueError("The slice in the {0} direction will remove "
                                 "all elements.  If you want a single-channel "
                                 "slice, you need a different approach."
                                 .format(xx))

        slices = [slice(limit_dict[xx+'lo'], limit_dict[xx+'hi'])
                  for xx in 'zyx']
        slices = tuple(slices)

        log.debug('slices: {0}'.format(slices))

        return self[slices]

    def subcube_from_ds9region(self, ds9_region, allow_empty=False):
        """
        Extract a masked subcube from a ds9 region
        (only functions on celestial dimensions)

        Parameters
        ----------
        ds9_region: str
            The DS9 region(s) to extract
        allow_empty: bool
            If this is False, an exception will be raised if the region
            contains no overlap with the cube
        """
        import regions

        if isinstance(ds9_region, six.string_types):
            region_list = regions.DS9Parser(ds9_region).shapes.to_regions()
        else:
            raise TypeError("{0} should be a DS9 string".format(ds9_region))

        return self.subcube_from_regions(region_list, allow_empty)

    def subcube_from_crtfregion(self, crtf_region, allow_empty=False):
        """
        Extract a masked subcube from a CRTF region.

        Parameters
        ----------
        crtf_region: str
            The CRTF region(s) string to extract
        allow_empty: bool
            If this is False, an exception will be raised if the region
            contains no overlap with the cube
        """
        import regions

        if isinstance(crtf_region, six.string_types):
            region_list = regions.CRTFParser(crtf_region).shapes.to_regions()
        else:
            raise TypeError("{0} should be a CRTF string".format(crtf_region))

        return self.subcube_from_regions(region_list, allow_empty)

    def subcube_from_regions(self, region_list, allow_empty=False):
        """
        Extract a masked subcube from a list of ``regions.Region`` object
        (only functions on celestial dimensions)

        Parameters
        ----------
        region_list: ``regions.Region`` list
            The region(s) to extract
        allow_empty: bool, optional
            If this is False, an exception will be raised if the region
            contains no overlap with the cube. Default is False.
        """
        import regions

        # Convert every region to a `regions.PixelRegion` object.
        regs = []
        for x in region_list:
            if isinstance(x, regions.SkyRegion):
                regs.append(x.to_pixel(self.wcs.celestial))
            elif isinstance(x, regions.PixelRegion):
                regs.append(x)
            else:
                raise TypeError("'{}' should be `regions.Region` object".format(x))

        # List of regions are converted to a `regions.CompoundPixelRegion` object.
        compound_region = _regionlist_to_single_region(regs)

        # Compound mask of all the regions.
        mask = compound_region.to_mask()

        # Collecting frequency/velocity range, velocity type and rest frequency
        # of each region.
        ranges = [x.meta.get('range', None) for x in regs]
        veltypes = [x.meta.get('veltype', None) for x in regs]
        restfreqs = [x.meta.get('restfreq', None) for x in regs]

        xlo, xhi, ylo, yhi = mask.bbox.ixmin, mask.bbox.ixmax, mask.bbox.iymin, mask.bbox.iymax

        # Negative indices will do bad things, like wrap around the cube
        # If xhi/yhi are negative, there is not overlap
        if (xhi < 0) or (yhi < 0):
            raise ValueError("Region is outside of cube.")

        if xlo < 0:
            xlo = 0
        if ylo < 0:
            ylo = 0

        # If None, then the whole spectral range of the cube is selected.
        if None in ranges:
            subcube = self.subcube(xlo=xlo, ylo=ylo, xhi=xhi, yhi=yhi)
        else:
            ranges = self._velocity_freq_conversion_regions(ranges, veltypes, restfreqs)
            zlo = min([x[0] for x in ranges])
            zhi = max([x[1] for x in ranges])
            slab = self.spectral_slab(zlo, zhi)
            subcube = slab.subcube(xlo=xlo, ylo=ylo, xhi=xhi, yhi=yhi)

        if any(dim == 0 for dim in subcube.shape):
            if allow_empty:
                warnings.warn("The derived subset is empty: the region does not"
                              " overlap with the cube (but allow_empty=True).")
            else:
                raise ValueError("The derived subset is empty: the region does not"
                                 " overlap with the cube.")

        # cropping the mask from top left corner so that it fits the subcube.
        maskarray = mask.data[:subcube.shape[1], :subcube.shape[2]].astype('bool')

        masked_subcube = subcube.with_mask(BooleanArrayMask(maskarray, subcube.wcs, shape=subcube.shape))
        # by using ceil / floor above, we potentially introduced a NaN buffer
        # that we can now crop out
        return masked_subcube.minimal_subcube(spatial_only=True)

    def _velocity_freq_conversion_regions(self, ranges, veltypes, restfreqs):
        """
        Makes the spectral range of the regions compatible with the spectral
        convention of the cube.

        ranges: `~astropy.units.Quantity` object
            List of range(a list of max and min limits on the spectral axis) of
            each ``regions.Region`` object.
        veltypes: List of `str`
            It contains list of velocity convention that each region is following.
            The string should be a combination of the following elements:
            {'RADIO' | 'OPTICAL' | 'Z' | 'BETA' | 'GAMMA' | 'RELATIVISTIC' | None}
            An element can be `None` if veltype of the region is unknown and is
            assumed to take that of the cube.
        restfreqs: List of `~astropy.units.Quantity`
            It contains the rest frequency of each region.
        """
        header = self.wcs.to_header()

        # Obtaining rest frequency of the cube in GHz.
        restfreq_cube = get_rest_value_from_wcs(self.wcs).to("GHz",
                                                           equivalencies=u.spectral())

        CTYPE3 = header['CTYPE3']

        veltype_cube = determine_vconv_from_ctype(CTYPE3)

        veltype_equivalencies = dict(RADIO=u.doppler_radio,
                                     OPTICAL=u.doppler_optical,
                                     Z=doppler_z,
                                     BETA=doppler_beta,
                                     GAMMA=doppler_gamma,
                                     RELATIVISTIC=u.doppler_relativistic
                                     )

        final_ranges = []

        for range, veltype, restfreq in zip(ranges, veltypes, restfreqs):

            if restfreq is None:
                restfreq = restfreq_cube

            restfreq = restfreq.to("GHz", equivalencies=u.spectral())

            if veltype not in veltype_equivalencies and veltype is not None:
                raise ValueError("Spectral Cube doesn't support {} this type of"
                                 "velocity".format(veltype))

            veltype = veltype_equivalencies.get(veltype, veltype_cube)

            # Because there is chance that the veltype  and rest frequency
            # of the region may not be the same as that of cube, we convert it
            # to frequency and then convert to the spectral unit of the cube.
            freq_range = (u.Quantity(range).to("GHz",
                          equivalencies=veltype(restfreq)))

            final_ranges.append(freq_range.to(header['CUNIT3'],
                                equivalencies=veltype_cube(restfreq_cube)))

        return final_ranges

    def _val_to_own_unit(self, value, operation='compare', tofrom='to',
                         keepunit=False):
        """
        Given a value, check if it has a unit.  If it does, convert to the
        cube's unit.  If it doesn't, raise an exception.
        """
        if isinstance(value, SpectralCube):
            if self.unit.is_equivalent(value.unit):
                return value
            else:
                return value.to(self.unit)
        elif hasattr(value, 'unit'):
            if keepunit:
                return value.to(self.unit)
            else:
                return value.to(self.unit).value
        else:
            raise ValueError("Can only {operation} cube objects {tofrom}"
                             " SpectralCubes or Quantities with "
                             "a unit attribute."
                             .format(operation=operation, tofrom=tofrom))

    def __gt__(self, value):
        """
        Return a LazyMask representing the inequality

        Parameters
        ----------
        value : number
            The threshold
        """
        value = self._val_to_own_unit(value)
        return LazyComparisonMask(operator.gt, value, data=self._data, wcs=self._wcs)

    def __ge__(self, value):
        value = self._val_to_own_unit(value)
        return LazyComparisonMask(operator.ge, value, data=self._data, wcs=self._wcs)

    def __le__(self, value):
        value = self._val_to_own_unit(value)
        return LazyComparisonMask(operator.le, value, data=self._data, wcs=self._wcs)

    def __lt__(self, value):
        value = self._val_to_own_unit(value)
        return LazyComparisonMask(operator.lt, value, data=self._data, wcs=self._wcs)

    def __eq__(self, value):
        value = self._val_to_own_unit(value)
        return LazyComparisonMask(operator.eq, value, data=self._data, wcs=self._wcs)

    def __hash__(self):
        return id(self)

    def __ne__(self, value):
        value = self._val_to_own_unit(value)
        return LazyComparisonMask(operator.ne, value, data=self._data, wcs=self._wcs)

    def __add__(self, value):
        if isinstance(value, SpectralCube):
            return self._cube_on_cube_operation(operator.add, value)
        else:
            value = self._val_to_own_unit(value, operation='add', tofrom='from',
                                          keepunit=True)
            return self._apply_everywhere(operator.add, value)

    def __sub__(self, value):
        if isinstance(value, SpectralCube):
            return self._cube_on_cube_operation(operator.sub, value)
        else:
            value = self._val_to_own_unit(value, operation='subtract',
                                          tofrom='from', keepunit=True)
            return self._apply_everywhere(operator.sub, value)

    def __mul__(self, value):
        if isinstance(value, SpectralCube):
            return self._cube_on_cube_operation(operator.mul, value)
        else:
            return self._apply_everywhere(operator.mul, value)

    def __truediv__(self, value):
        return self.__div__(value)

    def __div__(self, value):
        if isinstance(value, SpectralCube):
            return self._cube_on_cube_operation(operator.truediv, value)
        else:
            return self._apply_everywhere(operator.truediv, value)

    def __pow__(self, value):
        if isinstance(value, SpectralCube):
            return self._cube_on_cube_operation(operator.pow, value)
        else:
            return self._apply_everywhere(operator.pow, value)

    def to_yt(self, spectral_factor=1.0, nprocs=None, **kwargs):
        """
        Convert a spectral cube to a yt object that can be further analyzed in
        yt.

        Parameters
        ----------
        spectral_factor : float, optional
            Factor by which to stretch the spectral axis. If set to 1, one pixel
            in spectral coordinates is equivalent to one pixel in spatial
            coordinates.

        If using yt 3.0 or later, additional keyword arguments will be passed
        onto yt's ``FITSDataset`` constructor. See the yt documentation
        (http://yt-project.org/docs/3.0/examining/loading_data.html?#fits-data)
        for details on options for reading FITS data.
        """

        import yt

        if (('dev' in yt.__version__) or
            (LooseVersion(yt.__version__) >= LooseVersion('3.0'))):

            # yt has updated their FITS data set so that only the SpectralCube
            # variant takes spectral_factor
            try:
                from yt.frontends.fits.api import SpectralCubeFITSDataset as FITSDataset
            except ImportError:
                from yt.frontends.fits.api import FITSDataset
            from yt.units.unit_object import UnitParseError

            hdu = PrimaryHDU(self._get_filled_data(fill=0.),
                             header=self.wcs.to_header())

            units = str(self.unit.to_string())

            hdu.header["BUNIT"] = units
            hdu.header["BTYPE"] = "flux"

            ds = FITSDataset(hdu, nprocs=nprocs,
                             spectral_factor=spectral_factor, **kwargs)

            # Check to make sure the units are legit

            try:
                ds.quan(1.0,units)
            except UnitParseError:
                raise RuntimeError("The unit %s was not parsed by yt. " % units+
                                   "Check to make sure it is correct.")

        else:

            from yt.mods import load_uniform_grid

            data = {'flux': self._get_filled_data(fill=0.).transpose()}

            nz, ny, nx = self.shape

            if nprocs is None:
                nprocs = 1

            bbox = np.array([[0.5,float(nx)+0.5],
                             [0.5,float(ny)+0.5],
                             [0.5,spectral_factor*float(nz)+0.5]])

            ds = load_uniform_grid(data, [nx,ny,nz], 1., bbox=bbox,
                                   nprocs=nprocs, periodicity=(False, False,
                                                               False))

        return ytCube(self, ds, spectral_factor=spectral_factor)

    def to_glue(self, name=None, glue_app=None, dataset=None, start_gui=True):
        """
        Send data to a new or existing Glue application

        Parameters
        ----------
        name : str or None
            The name of the dataset within Glue.  If None, defaults to
            'SpectralCube'.  If a dataset with the given name already exists,
            a new dataset with "_" appended will be added instead.
        glue_app : GlueApplication or None
            A glue application to send the data to.  If this is not specified,
            a new glue application will be started if one does not already
            exist for this cube.  Otherwise, the data will be sent to the
            existing glue application, `self._glue_app`.
        dataset : glue.core.Data or None
            An existing Data object to add the cube to.  This is a good way
            to compare cubes with the same dimensions.  Supercedes ``glue_app``
        start_gui : bool
            Start the GUI when this is run.  Set to `False` for testing.
        """
        if name is None:
            name = 'SpectralCube'

        from glue.app.qt import GlueApplication
        from glue.core import DataCollection, Data
        from glue.core.coordinates import coordinates_from_header
        try:
            from glue.viewers.image.qt.data_viewer import ImageViewer
        except ImportError:
            from glue.viewers.image.qt.viewer_widget import ImageWidget as ImageViewer

        if dataset is not None:
            if name in [d.label for d in dataset.components]:
                name = name+"_"
            dataset[name] = self

        else:
            result = Data(label=name)
            result.coords = coordinates_from_header(self.header)

            result.add_component(self, name)

            if glue_app is None:
                if hasattr(self,'_glue_app'):
                    glue_app = self._glue_app
                else:
                    # Start a new glue session.  This will quit when done.
                    # I don't think the return statement is ever reached, based on
                    # past attempts [@ChrisBeaumont - chime in here if you'd like]
                    dc = DataCollection([result])

                    #start Glue
                    ga = self._glue_app = GlueApplication(dc)
                    self._glue_viewer = ga.new_data_viewer(ImageViewer,
                                                           data=result)

                    if start_gui:
                        self._glue_app.start()

                    return self._glue_app

            glue_app.add_datasets(self._glue_app.data_collection, result)


    def to_pvextractor(self):
        """
        Open the cube in a quick viewer written in matplotlib that allows you
        to create PV extractions within the GUI
        """
        from pvextractor.gui import PVSlicer

        return PVSlicer(self)

    def to_ds9(self, ds9id=None, newframe=False):
        """
        Send the data to ds9 (this will create a copy in memory)

        Parameters
        ----------
        ds9id: None or string
            The DS9 session ID.  If 'None', a new one will be created.
            To find your ds9 session ID, open the ds9 menu option
            File:XPA:Information and look for the XPA_METHOD string, e.g.
            ``XPA_METHOD:  86ab2314:60063``.  You would then calll this
            function as ``cube.to_ds9('86ab2314:60063')``
        newframe: bool
            Send the cube to a new frame or to the current frame?
        """
        try:
            import ds9
        except ImportError:
            import pyds9 as ds9

        if ds9id is None:
            dd = ds9.DS9(start=True)
        else:
            dd = ds9.DS9(target=ds9id, start=False)

        if newframe:
            dd.set('frame new')

        dd.set_pyfits(self.hdulist)

        return dd


    @property
    def header(self):
        log.debug("Creating header")

        header = super(BaseSpectralCube, self).header

        # Preserve the cube's spectral units
        # (if CUNIT3 is not in the header, it is whatever that type's default unit is)
        if 'CUNIT3' in header and self._spectral_unit != u.Unit(header['CUNIT3']):
            header['CDELT3'] *= self._spectral_scale
            header['CRVAL3'] *= self._spectral_scale
            header['CUNIT3'] = self._spectral_unit.to_string(format='FITS')

        return header

    @property
    def hdu(self):
        """
        HDU version of self
        """
        log.debug("Creating HDU")
        hdu = PrimaryHDU(self.filled_data[:].value, header=self.header)
        return hdu

    @property
    def hdulist(self):
        return HDUList(self.hdu)

    @warn_slow
    def to(self, unit, equivalencies=()):
        """
        Return the cube converted to the given unit (assuming it is equivalent).
        If conversion was required, this will be a copy, otherwise it will
        """

        if not isinstance(unit, u.Unit):
            unit = u.Unit(unit)

        if unit == self.unit:
            # No copying
            return self

        if self.unit.is_equivalent(u.Jy/u.beam):
            # replace "beam" with the actual beam
            if not hasattr(self, 'beam') or self._beam is None:
                raise ValueError("To convert cubes with Jy/beam units, "
                                 "the cube needs to have a beam defined.")
            brightness_unit = self.unit * u.beam

            # create a beam equivalency for brightness temperature
            bmequiv = self.beam.jtok_equiv(self.with_spectral_unit(u.Hz).spectral_axis)
            factor = brightness_unit.to(unit,
                                        equivalencies=bmequiv+list(equivalencies))
        else:
            # scaling factor
            factor = self.unit.to(unit, equivalencies=equivalencies)

        # special case: array in equivalencies
        # (I don't think this should have to be special cased, but I don't know
        # how to manipulate broadcasting rules any other way)
        if hasattr(factor, '__len__') and len(factor) == len(self):
            return self._new_cube_with(data=self._data*factor[:,None,None],
                                       unit=unit)
        else:
            return self._new_cube_with(data=self._data*factor,
                                       unit=unit)


    def find_lines(self, velocity_offset=None, velocity_convention=None,
                   rest_value=None, **kwargs):
        """
        Using astroquery's splatalogue interface, search for lines within the
        spectral band.  See `astroquery.splatalogue.Splatalogue` for
        information on keyword arguments

        Parameters
        ----------
        velocity_offset : u.km/u.s equivalent
            An offset by which the spectral axis should be shifted before
            searching splatalogue.  This value will be *added* to the velocity,
            so if you want to redshift a spectrum, make this value positive,
            and if you want to un-redshift it, make this value negative.
        velocity_convention : 'radio', 'optical', 'relativistic'
            The doppler convention to pass to `with_spectral_unit`
        rest_value : u.GHz equivalent
            The rest frequency (or wavelength or energy) to be passed to
            `with_spectral_unit`
        """
        warnings.warn("The line-finding routine is experimental.  Please "
                      "report bugs on the Issues page: "
                      "https://github.com/radio-astro-tools/spectral-cube/issues",
                      ExperimentalImplementationWarning
                     )
        from astroquery.splatalogue import Splatalogue
        if velocity_convention in DOPPLER_CONVENTIONS:
            velocity_convention = DOPPLER_CONVENTIONS[velocity_convention]
        if velocity_offset is not None:
            newspecaxis = self.with_spectral_unit(u.km/u.s,
                                                  velocity_convention=velocity_convention,
                                                  rest_value=rest_value).spectral_axis
            spectral_axis = (newspecaxis + velocity_offset).to(u.GHz,
                                                               velocity_convention(rest_value))
        else:
            spectral_axis = self.spectral_axis.to(u.GHz)

        numin,numax = spectral_axis.min(), spectral_axis.max()

        log.log(19, "Min/max frequency: {0},{1}".format(numin, numax))

        result = Splatalogue.query_lines(numin, numax, **kwargs)

        return result

    @warn_slow
    def reproject(self, header, order='bilinear', use_memmap=False,
                  filled=True):
        """
        Spatially reproject the cube into a new header.  Fills the data with
        the cube's ``fill_value`` to replace bad values before reprojection.

        If you want to reproject a cube both spatially and spectrally, you need
        to use `spectral_interpolate` as well.

        .. warning::
            The current implementation of ``reproject`` requires that the whole
            cube be loaded into memory.  Issue #506 notes that this is a
            problem, and it is on our to-do list to fix.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            A header specifying a cube in valid WCS
        order : int or str, optional
            The order of the interpolation (if ``mode`` is set to
            ``'interpolation'``). This can be either one of the following
            strings:

                * 'nearest-neighbor'
                * 'bilinear'
                * 'biquadratic'
                * 'bicubic'

            or an integer. A value of ``0`` indicates nearest neighbor
            interpolation.
        use_memmap : bool
            If specified, a memory mapped temporary file on disk will be
            written to rather than storing the intermediate spectra in memory.
        filled : bool
            Fill the masked values with the cube's fill value before
            reprojection?  Note that setting ``filled=False`` will use the raw
            data array, which can be a workaround that prevents loading large
            data into memory.
        """

        try:
            from reproject.version import version
        except ImportError:
            raise ImportError("Requires the reproject package to be"
                              " installed.")

        # Need version > 0.2 to work with cubes, >= 0.5 for memmap
        from distutils.version import LooseVersion
        if LooseVersion(version) < "0.5":
            raise Warning("Requires version >=0.5 of reproject. The current "
                          "version is: {}".format(version))
        elif LooseVersion(version) >= "0.6":
            reproj_kwargs = {}
        else:
            reproj_kwargs = {'independent_celestial_slices': True}

        from reproject import reproject_interp

        # TODO: Find the minimal subcube that contains the header and only reproject that
        # (see FITS_tools.regrid_cube for a guide on how to do this)

        newwcs = wcs.WCS(header)
        shape_out = tuple([header['NAXIS{0}'.format(i + 1)] for i in
                           range(header['NAXIS'])][::-1])

        if filled:
            data = self.unitless_filled_data[:]
        else:
            data = self._data

        if use_memmap:
            if data.dtype.itemsize not in (4,8):
                raise ValueError("Data must be float32 or float64 to be "
                                 "reprojected.  Other data types need some "
                                 "kind of additional memory handling.")
            # note: requires reproject from December 2018 or later
            outarray = np.memmap(filename='output.np', mode='w+',
                                 shape=tuple(shape_out),
                                 dtype='float64' if data.dtype.itemsize == 8 else 'float32')
        else:
            outarray = None

        newcube, newcube_valid = reproject_interp((data,
                                                   self.header),
                                                  newwcs,
                                                  output_array=outarray,
                                                  shape_out=shape_out,
                                                  order=order,
                                                  **reproj_kwargs)

        return self._new_cube_with(data=newcube,
                                   wcs=newwcs,
                                   mask=BooleanArrayMask(newcube_valid.astype('bool'),
                                                         newwcs),
                                   meta=self.meta,
                                  )


    @parallel_docstring
    def spatial_smooth_median(self, ksize, update_function=None, **kwargs):
        """
        Smooth the image in each spatial-spatial plane of the cube using a median filter.

        Parameters
        ----------
        ksize : int
            Size of the median filter (scipy.ndimage.filters.median_filter)
        update_function : method
            Method that is called to update an external progressbar
            If provided, it disables the default `astropy.utils.console.ProgressBar`
        kwargs : dict
            Passed to the convolve function
        """
        if not scipyOK:
            raise ImportError("Scipy could not be imported: this function won't work.")

        def _msmooth_image(im, **kwargs):
            return ndimage.filters.median_filter(im, size=ksize, **kwargs)

        newcube = self.apply_function_parallel_spatial(_msmooth_image,
                                                       **kwargs)

        return newcube

    @parallel_docstring
    def spatial_smooth(self, kernel,
                       convolve=convolution.convolve,
                       **kwargs):
        """
        Smooth the image in each spatial-spatial plane of the cube.

        Parameters
        ----------
        kernel : `~astropy.convolution.Kernel2D`
            A 2D kernel from astropy
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`
        kwargs : dict
            Passed to the convolve function
        """

        def _gsmooth_image(img, **kwargs):
            """
            Helper function to smooth an image
            """
            return convolve(img, kernel, normalize_kernel=True, **kwargs)

        newcube = self.apply_function_parallel_spatial(_gsmooth_image,
                                                       **kwargs)

        return newcube

    @parallel_docstring
    def spectral_smooth_median(self, ksize,
                               use_memmap=True,
                               verbose=0,
                               num_cores=None,
                               **kwargs):
        """
        Smooth the cube along the spectral dimension

        Parameters
        ----------
        ksize : int
            Size of the median filter (scipy.ndimage.filters.median_filter)
        verbose : int
            Verbosity level to pass to joblib
        kwargs : dict
            Not used at the moment.
        """

        if not scipyOK:
            raise ImportError("Scipy could not be imported: this function won't work.")

        return self.apply_function_parallel_spectral(ndimage.filters.median_filter,
                                                     size=ksize,
                                                     verbose=verbose,
                                                     num_cores=num_cores,
                                                     use_memmap=use_memmap,
                                                     **kwargs)

    def _apply_function_parallel_base(self,
                                      iteration_data,
                                      function,
                                      applicator,
                                      num_cores=None,
                                      verbose=0,
                                      use_memmap=True,
                                      parallel=False,
                                      use_dask=False,
                                      memmap_dir=None,
                                      update_function=None,
                                      number_of_operations=None,
                                      **kwargs
                                     ):
        """
        Apply a function in parallel using the ``applicator`` function.  The
        function will be performed on data with masked values replaced with the
        cube's fill value.

        Parameters
        ----------
        iteration_data : generator
            The data to be iterated over in the format expected by ``applicator``
        function : function
            The function to apply in the spectral dimension.  It must take
            two arguments: an array representing a spectrum and a boolean array
            representing the mask.  It may also accept ``**kwargs``.  The
            function must return an object with the same shape as the input
            spectrum.
        applicator : function
            Either ``_apply_spatial_function`` or ``_apply_spectral_function``,
            a tool to handle the iteration data and send it to the ``function``
            appropriately.
        num_cores : int or None
            The number of cores to use if running in parallel.  Should be >1 if
            ``parallel==True`` and cannot be >1 if ``parallel==False``
        verbose : int
            Verbosity level to pass to joblib
        use_memmap : bool
            If specified, a memory mapped temporary file on disk will be
            written to rather than storing the intermediate spectra in memory.
        parallel : bool
            If set to ``False``, will force the use of a single thread instead
            of using ``joblib``.
        update_function : function
            A callback function to call on each iteration of the application.
            It should not accept any arguments.  For example, this can be
            ``Progressbar.update`` or some function that prints a status
            report.  The function *must* be picklable if ``parallel==True``.
        kwargs : dict
            Passed to ``function``
        """

        if use_memmap:
            ntf = tempfile.NamedTemporaryFile(dir=memmap_dir)
            outcube = np.memmap(ntf.name, mode='w+', shape=self.shape, dtype=np.float)
        else:
            if self._is_huge and not self.allow_huge_operations:
                raise ValueError("Applying a function without ``use_memmap`` "
                                 "requires loading the whole array into "
                                 "memory *twice*, which can overload the "
                                 "machine's memory for large cubes.  Either "
                                 "set ``use_memmap=True`` or set "
                                 "``cube.allow_huge_operations=True`` to "
                                 "override this restriction.")
            outcube = np.empty(shape=self.shape, dtype=np.float)

        if num_cores == 1 and parallel:
            warnings.warn("parallel=True was specified but num_cores=1. "
                          "Joblib will be used to run the task with a "
                          "single thread.")
        elif num_cores is not None and num_cores > 1 and not parallel:
            raise ValueError("parallel execution was not requested, but "
                             "multiple cores were: these are incompatible "
                             "options.  Either specify num_cores=1 or "
                             "parallel=True")

        if use_dask:
            import dask
            import dask.array
            import dask.distributed
            try:
                client = dask.distributed.get_client()
            except ValueError:
                client = dask.distributed.Client()

            out_target = ntf.name if use_memmap else outcube

            applicator_calls = [dask.delayed(applicator)(arg,
                                                         out_target,
                                                         dask.delayed(function),
                                                         shape=self.shape,
                                                         return_vals=not use_memmap,
                                                         **kwargs)
                                for arg in iteration_data]

            if use_memmap:
                # compresult will be [None] * ncalls because return_vals=False
                compresult = client.compute(applicator_calls, sync=True)
            else:
                # I don't know why you can't use from_delayed:
                #arr = dask.array.from_delayed(dask.delayed(applicator_calls),
                #                              shape=self.shape,
                #                              dtype=self.dtype
                #                             )

                arr = dask.array.from_array(client.compute(applicator_calls,
                                                           sync=True))

                # client.compute will produce something with a wrong shape,
                # where one dimension is n_processors
                if applicator is _apply_spectral_function:
                    # there must be a better way to test this?
                    arr = arr.T.reshape(self.shape)
                outcube[:] = arr.compute()

        elif parallel and use_memmap:

            # it is not possible to run joblib parallelization without memmap
            try:
                import joblib
                from joblib._parallel_backends import MultiprocessingBackend
                from joblib import register_parallel_backend, parallel_backend
                from joblib import Parallel, delayed

                if update_function is not None:

                    # https://stackoverflow.com/questions/38483874/intermediate-results-from-joblib
                    class MultiCallback:
                        def __init__(self, *callbacks):
                            self.callbacks = [cb for cb in callbacks if cb]

                        def __call__(self, out):
                            for cb in self.callbacks:
                                cb(out)

                    class Callback_Backend(MultiprocessingBackend):
                        def callback(self, result):
                            update_function()

                        # Overload apply_async and set callback=self.callback
                        def apply_async(self, func, callback=None):
                            cbs = MultiCallback(callback, self.callback)
                            return super().apply_async(func, cbs)

                    joblib.register_parallel_backend('custom',
                                                     Callback_Backend,
                                                     make_default=True)

                Parallel(n_jobs=num_cores,
                         verbose=verbose,
                         max_nbytes=None)(delayed(applicator)(arg, outcube,
                                                              function,
                                                              **kwargs)
                                          for arg in iteration_data)
            except ImportError:
                if num_cores is not None and num_cores > 1:
                    warnings.warn("Could not import joblib.  Will run in serial.",
                                  warnings.ImportWarning)
                parallel = False

        # this isn't an else statement because we want to catch the case where
        # the above clause fails on ImportError
        if not parallel or not use_memmap:
            if update_function is not None:
                pbu = update_function
            elif verbose > 0:
                progressbar = ProgressBar(number_of_operations)
                pbu = progressbar.update
            else:
                pbu = object

            for arg in iteration_data:
                applicator(arg, outcube, function, **kwargs)
                pbu()


        # TODO: do something about the mask?
        newcube = self._new_cube_with(data=outcube, wcs=self.wcs,
                                      mask=self.mask, meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube

    def apply_function_parallel_spatial(self,
                                        function,
                                        num_cores=None,
                                        verbose=0,
                                        use_memmap=True,
                                        parallel=True,
                                        use_dask=False,
                                        **kwargs
                                       ):
        """
        Apply a function in parallel along the spatial dimension.  The
        function will be performed on data with masked values replaced with the
        cube's fill value.

        Parameters
        ----------
        function : function
            The function to apply in the spatial dimension.  It must take
            two arguments: an array representing an image and a boolean array
            representing the mask.  It may also accept ``**kwargs``.  The
            function must return an object with the same shape as the input
            spectrum.
        num_cores : int or None
            The number of cores to use if running in parallel
        verbose : int
            Verbosity level to pass to joblib
        use_memmap : bool
            If specified, a memory mapped temporary file on disk will be
            written to rather than storing the intermediate spectra in memory.
        parallel : bool
            If set to ``False``, will force the use of a single core without
            using ``joblib``.
        kwargs : dict
            Passed to ``function``
        """
        shape = self.shape

        data = self.unitless_filled_data

        def data_getter(zz):
            return data[zz, :, :]
        def mask_getter(zz):
            return self.mask.include(view=(zz, slice(None), slice(None)))

        if use_dask:
            import dask
            data_getter = dask.delayed(data_getter)
            mask_getter = dask.delayed(mask_getter)


        # 'images' is a generator
        # the boolean check will skip the function for bad spectra
        images = ((data_getter,
                   mask_getter,
                   ii,
                   )
                  for ii in range(shape[0]))

        return self._apply_function_parallel_base(images, function,
                                                  applicator=_apply_spatial_function,
                                                  verbose=verbose,
                                                  parallel=parallel,
                                                  num_cores=num_cores,
                                                  use_memmap=use_memmap,
                                                  use_dask=use_dask,
                                                  number_of_operations=self.shape[0],
                                                  **kwargs)

    def apply_function_parallel_spectral(self,
                                         function,
                                         num_cores=None,
                                         verbose=0,
                                         use_memmap=True,
                                         parallel=True,
                                         use_dask=False,
                                         **kwargs
                                        ):
        """
        Apply a function in parallel along the spectral dimension.  The
        function will be performed on data with masked values replaced with the
        cube's fill value.

        Parameters
        ----------
        function : function
            The function to apply in the spectral dimension.  It must take
            two arguments: an array representing a spectrum and a boolean array
            representing the mask.  It may also accept ``**kwargs``.  The
            function must return an object with the same shape as the input
            spectrum.
        num_cores : int or None
            The number of cores to use if running in parallel
        verbose : int
            Verbosity level to pass to joblib
        use_memmap : bool
            If specified, a memory mapped temporary file on disk will be
            written to rather than storing the intermediate spectra in memory.
        parallel : bool
            If set to ``False``, will force the use of a single core without
            using ``joblib``.
        kwargs : dict
            Passed to ``function``
        """
        shape = self.shape

        data = self.unitless_filled_data

        def data_getter(xx, yy):
            return data[:, yy, xx]
        def mask_getter(xx, yy):
            return self.mask.include(view=(slice(None), yy, xx))

        if use_dask:
            import dask
            data_getter = dask.delayed(data_getter)
            mask_getter = dask.delayed(mask_getter)

        # 'spectra' is a generator
        # the boolean check will skip the function for bad spectra
        # TODO: should spatial good/bad be cached?
        spectra = ((data_getter,
                    mask_getter,
                    ii, jj,
                   )
                   for jj in range(shape[1])
                   for ii in range(shape[2]))

        return self._apply_function_parallel_base(iteration_data=spectra,
                                                  function=function,
                                                  applicator=_apply_spectral_function,
                                                  use_memmap=use_memmap,
                                                  parallel=parallel,
                                                  verbose=verbose,
                                                  num_cores=num_cores,
                                                  use_dask=use_dask,
                                                  number_of_operations=self.shape[1]*self.shape[2],
                                                  **kwargs
                                                 )

    @parallel_docstring
    def sigma_clip_spectrally(self, threshold, verbose=0, use_memmap=True,
                              num_cores=None, **kwargs):
        """
        Run astropy's sigma clipper along the spectral axis, converting all bad
        (excluded) values to NaN.

        Parameters
        ----------
        threshold : float
            The ``sigma`` parameter in `astropy.stats.sigma_clip`, which refers
            to the number of sigma above which to cut.
        verbose : int
            Verbosity level to pass to joblib

        """

        return self.apply_function_parallel_spectral(stats.sigma_clip,
                                                     sigma=threshold,
                                                     axis=0, # changes behavior of sigmaclip
                                                     num_cores=num_cores,
                                                     use_memmap=use_memmap,
                                                     verbose=verbose,
                                                     **kwargs)

    @parallel_docstring
    def spectral_smooth(self, kernel,
                        convolve=convolution.convolve,
                        verbose=0,
                        use_memmap=True,
                        num_cores=None,
                        **kwargs):
        """
        Smooth the cube along the spectral dimension

        Note that the mask is left unchanged in this operation.

        Parameters
        ----------
        kernel : `~astropy.convolution.Kernel1D`
            A 1D kernel from astropy
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`
        verbose : int
            Verbosity level to pass to joblib
        kwargs : dict
            Passed to the convolve function
        """

        if isinstance(kernel.array, u.Quantity):
            raise u.UnitsError("The convolution kernel should be defined "
                               "without a unit.")

        return self.apply_function_parallel_spectral(convolve,
                                                     kernel=kernel,
                                                     normalize_kernel=True,
                                                     num_cores=num_cores,
                                                     use_memmap=use_memmap,
                                                     verbose=verbose,
                                                     **kwargs)

    def spectral_interpolate(self, spectral_grid,
                             suppress_smooth_warning=False,
                             fill_value=None,
                             update_function=None):
        """Resample the cube spectrally onto a specific grid

        Parameters
        ----------
        spectral_grid : array
            An array of the spectral positions to regrid onto
        suppress_smooth_warning : bool
            If disabled, a warning will be raised when interpolating onto a
            grid that does not nyquist sample the existing grid.  Disable this
            if you have already appropriately smoothed the data.
        fill_value : float
            Value for extrapolated spectral values that lie outside of
            the spectral range defined in the original data.  The
            default is to use the nearest spectral channel in the
            cube.
        update_function : method
            Method that is called to update an external progressbar
            If provided, it disables the default `astropy.utils.console.ProgressBar`

        Returns
        -------
        cube : SpectralCube

        """

        inaxis = self.spectral_axis.to(spectral_grid.unit)

        indiff = np.mean(np.diff(inaxis))
        outdiff = np.mean(np.diff(spectral_grid))

        # account for reversed axes
        if outdiff < 0:
            spectral_grid = spectral_grid[::-1]
            outdiff = np.mean(np.diff(spectral_grid))
            outslice = slice(None, None, -1)
        else:
            outslice = slice(None, None, 1)

        cubedata = self.filled_data
        specslice = slice(None) if indiff >= 0 else slice(None, None, -1)
        inaxis = inaxis[specslice]
        indiff = np.mean(np.diff(inaxis))

        # insanity checks
        if indiff < 0 or outdiff < 0:
            raise ValueError("impossible.")

        assert np.all(np.diff(spectral_grid) > 0)
        assert np.all(np.diff(inaxis) > 0)

        np.testing.assert_allclose(np.diff(spectral_grid), outdiff,
                                   err_msg="Output grid must be linear")

        if outdiff > 2 * indiff and not suppress_smooth_warning:
            warnings.warn("Input grid has too small a spacing. The data should "
                          "be smoothed prior to resampling.",
                         SmoothingWarning
                         )


        newcube = np.empty([spectral_grid.size, self.shape[1], self.shape[2]],
                           dtype=cubedata[:1, 0, 0].dtype)
        newmask = np.empty([spectral_grid.size, self.shape[1], self.shape[2]],
                           dtype='bool')

        yy,xx = np.indices(self.shape[1:])
        if update_function is None:
            pb = ProgressBar(xx.size)
            update_function = pb.update

        for ix, iy in (zip(xx.flat, yy.flat)):
            mask = self.mask.include(view=(specslice, iy, ix))
            if any(mask):
                newcube[outslice,iy,ix] = \
                    np.interp(spectral_grid.value, inaxis.value,
                              cubedata[specslice,iy,ix].value,
                              left=fill_value, right=fill_value)
                if all(mask):
                    newmask[:,iy,ix] = True
                else:
                    interped = np.interp(spectral_grid.value,
                                         inaxis.value, mask) > 0
                    newmask[outslice,iy,ix] = interped
            else:
                newmask[:, iy, ix] = False
                newcube[:, iy, ix] = np.NaN

            update_function()

        newwcs = self.wcs.deepcopy()
        newwcs.wcs.crpix[2] = 1
        newwcs.wcs.crval[2] = spectral_grid[0].value if outslice.step > 0 \
            else spectral_grid[-1].value
        newwcs.wcs.cunit[2] = spectral_grid.unit.to_string('FITS')
        newwcs.wcs.cdelt[2] = outdiff.value if outslice.step > 0 \
            else -outdiff.value
        newwcs.wcs.set()

        newbmask = BooleanArrayMask(newmask, wcs=newwcs)

        newcube = self._new_cube_with(data=newcube, wcs=newwcs, mask=newbmask,
                                      meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube

    @warn_slow
    def convolve_to(self, beam, convolve=convolution.convolve_fft, update_function=None, **kwargs):
        """
        Convolve each channel in the cube to a specified beam

        .. warning::
            The current implementation of ``convolve_to`` creates an in-memory
            copy of the whole cube to store the convolved data.  Issue #506
            notes that this is a problem, and it is on our to-do list to fix.

        Parameters
        ----------
        beam : `radio_beam.Beam`
            The beam to convolve to
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`
        update_function : method
            Method that is called to update an external progressbar
            If provided, it disables the default `astropy.utils.console.ProgressBar`
        kwargs : dict
            Keyword arguments to pass to the convolution function

        Returns
        -------
        cube : `SpectralCube`
            A SpectralCube with a single ``beam``
        """

        # Check if the beams are the same.
        if beam == self.beam:
            warnings.warn("The given beam is identical to the current beam. "
                          "Skipping convolution.")
            return self

        pixscale = wcs.utils.proj_plane_pixel_area(self.wcs.celestial)**0.5*u.deg

        convolution_kernel = beam.deconvolve(self.beam).as_kernel(pixscale)

        def convfunc(img):
            return convolve(img, convolution_kernel, normalize_kernel=True,
                            **kwargs)

        newcube = self.apply_function_parallel_spatial(convfunc,
                                                       **kwargs).with_beam(beam)

        return newcube

    def mask_channels(self, goodchannels):
        """
        Helper function to mask out channels.  This function is equivalent to
        adding a mask with ``cube[view]`` where ``view`` is broadcastable to
        the cube shape, but it accepts 1D arrays that are not normally
        broadcastable.

        Parameters
        ----------
        goodchannels : array
            A 1D boolean array declaring which channels should be kept.

        Returns
        -------
        cube : `SpectralCube`
            A cube with the specified channels masked
        """
        goodchannels = np.asarray(goodchannels, dtype='bool')

        if goodchannels.ndim != 1:
            raise ValueError("goodchannels mask must be one-dimensional")
        if goodchannels.size != self.shape[0]:
            raise ValueError("goodchannels must have a length equal to the "
                             "cube's spectral dimension.")
        return self.with_mask(goodchannels[:,None,None])


    @warn_slow
    def downsample_axis(self, factor, axis, estimator=np.nanmean,
                        truncate=False, use_memmap=True, progressbar=True):
        """
        Downsample the cube by averaging over *factor* pixels along an axis.
        Crops right side if the shape is not a multiple of factor.

        The WCS will be 'downsampled' by the specified factor as well.
        If the downsample factor is odd, there will be an offset in the WCS.

        There is both an in-memory and a memory-mapped implementation; the
        default is to use the memory-mapped version.  Technically, the 'large
        data' warning doesn't apply when using the memory-mapped version, but
        the warning is still there anyway.

        Parameters
        ----------
        myarr : `~numpy.ndarray`
            The array to downsample
        factor : int
            The factor to downsample by
        axis : int
            The axis to downsample along
        estimator : function
            defaults to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
        truncate : bool
            Whether to truncate the last chunk or average over a smaller number.
            e.g., if you downsample [1,2,3,4] by a factor of 3, you could get either
            [2] or [2,4] if truncate is True or False, respectively.
        use_memmap : bool
            Use a memory map on disk to avoid loading the whole cube into memory
            (several times)?  If set, the warning about large cubes can be ignored
            (though you still have to override the warning)
        progressbar : bool
            Include a progress bar?  Only works with ``use_memmap=True``
        """
        def makeslice(startpoint,axis=axis,step=factor):
            # make empty slices
            view = [slice(None) for ii in range(self.ndim)]
            # then fill the appropriate slice
            view[axis] = slice(startpoint,None,step)
            return tuple(view)

        # size of the dimension of interest
        xs = self.shape[axis]

        if not use_memmap:
            if xs % int(factor) != 0:
                if truncate:
                    view = [slice(None) for ii in range(self.ndim)]
                    view[axis] = slice(None,xs-(xs % int(factor)))
                    view = tuple(view)
                    crarr = self.unitless_filled_data[view]
                    mask = self.mask[view].include()
                else:
                    extension_shape = list(self.shape)
                    extension_shape[axis] = (factor - xs % int(factor))
                    extension = np.empty(extension_shape) * np.nan
                    crarr = np.concatenate((self.unitless_filled_data[:],
                                            extension), axis=axis)
                    extension[:] = 0
                    mask = np.concatenate((self.mask.include(), extension), axis=axis)
            else:
                crarr = self.unitless_filled_data[:]
                mask = self.mask.include()

            # The extra braces here are crucial: We're adding an extra dimension so we
            # can average across it
            stacked_array = np.concatenate([[crarr[makeslice(ii)]]
                                            for ii in range(factor)])

            dsarr = estimator(stacked_array, axis=0)

            if not isinstance(mask, np.ndarray):
                raise TypeError("Mask is of wrong data type")

            stacked_mask = np.concatenate([[mask[makeslice(ii)]] for ii in
                                           range(factor)])

            mask = np.any(stacked_mask, axis=0)
        else:
            def makeslice_local(startpoint, axis=axis, nsteps=factor):
                # make empty slices
                view = [slice(None) for ii in range(self.ndim)]
                # then fill the appropriate slice
                view[axis] = slice(startpoint,startpoint+nsteps,1)
                return tuple(view)

            newshape = list(self.shape)
            newshape[axis] = (newshape[axis]//factor +
                              ((1-int(truncate)) * (xs % int(factor) != 0)))
            newshape = tuple(newshape)

            if progressbar:
                progressbar = ProgressBar
            else:
                progressbar = lambda x: x

            # Create a view that will add a blank newaxis at the right spot
            view_newaxis = [slice(None) for ii in range(self.ndim)]
            view_newaxis[axis] = None
            view_newaxis = tuple(view_newaxis)

            ntf = tempfile.NamedTemporaryFile()
            dsarr = np.memmap(ntf, mode='w+', shape=newshape, dtype=np.float)
            ntf2 = tempfile.NamedTemporaryFile()
            mask = np.memmap(ntf2, mode='w+', shape=newshape, dtype=np.bool)
            for ii in progressbar(range(newshape[axis])):
                view_fulldata = makeslice_local(ii*factor)
                view_newdata = makeslice_local(ii, nsteps=1)

                to_average = self.unitless_filled_data[view_fulldata]
                to_anyfy = self.mask[view_fulldata].include()

                dsarr[view_newdata] = estimator(to_average, axis)[view_newaxis]
                mask[view_newdata] = np.any(to_anyfy, axis).astype('bool')[view_newaxis]


        # the slice should just start at zero; we had factor//2 here earlier,
        # and that was an error that probably half-compensated for an error in
        # wcs_utils
        view = makeslice(0)
        newwcs = wcs_utils.slice_wcs(self.wcs, view, shape=self.shape)
        newwcs._naxis = list(self.shape)

        # this is an assertion to ensure that the WCS produced is valid
        # (this is basically a regression test for #442)
        assert newwcs[:, slice(None), slice(None)]
        assert len(newwcs._naxis) == 3


        return self._new_cube_with(data=dsarr, wcs=newwcs,
                                   mask=BooleanArrayMask(mask, wcs=newwcs))

    def plot_channel_maps(self, nx, ny, channels, contourkwargs={}, output_file=None,
                          fig=None, fig_smallest_dim_inches=8, decimals=3, zoom=1,
                          textcolor=None, cmap='gray_r', tighten=False,
                          textxloc=0.5, textyloc=0.9,
                          savefig_kwargs={}, **kwargs):
        """
        Make channel maps from a spectral cube

        Parameters
        ----------
        input_file : str
            Name of the input spectral cube
        nx, ny : int
            Number of sub-plots in the x and y direction
        channels : list
            List of channels to show
        cmap : str
            The name of a colormap to use for the ``imshow`` colors
        contourcolors : list
            A list of contour colors corresponding to the contour levels
        textcolor : None or str
            Color of the label text to overlay.  If ``None``, will be
            determined automatically.  If ``'notext'``, no text will be added.
        textxloc : float
        textyloc : float
            Text label X,Y-location in axis fraction units
        output_file : str
            Name of the matplotlib plot
        fig : matplotlib figure
            The figure object to plot onto.  Will be overridden to enforce a
            specific aspect ratio.
        fig_smallest_dim_inches : float
            The size of the smallest dimension (either width or height) of the
            figure in inches.  The other dimension will be selected based on
            the aspect ratio of the data: it cannot be a free parameter.
        decimals : int, optional
            Number of decimal places to show in spectral value
        zoom : int, optional
            How much to zoom in. In future versions of this function, the
            pointing center will be customizable.
        tighten : bool
            Call ``plt.tight_layout()`` after plotting?
        savefig_kwargs : dict
            Keyword arguments to pass to ``savefig`` (e.g.,
            ``bbox_inches='tight'``)
        kwargs : dict
            Passed to ``imshow``
        """

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        cmap = getattr(plt.cm, cmap)

        if len(channels) != nx * ny:
            raise ValueError("Number of channels should be equal to nx * ny")

        # Read in spectral cube and get spectral axis
        spectral_axis = self.spectral_axis

        sizey, sizex = self.shape[1:]
        cenx = sizex / 2.
        ceny = sizey / 2.

        aspect_ratio = self.shape[2]/float(self.shape[1])
        gridratio = ny / float(nx) * aspect_ratio
        if gridratio > 1:
            ysize = fig_smallest_dim_inches*gridratio
            xsize = fig_smallest_dim_inches
        else:
            xsize = fig_smallest_dim_inches*gridratio
            ysize = fig_smallest_dim_inches

        if fig is None:
            fig = plt.figure(figsize=(xsize, ysize))
        else:
            fig.set_figheight(ysize)
            fig.set_figwidth(xsize)
        # unclear if needed
        #fig.subplots_adjust(margin,margin,1.-margin,1.-margin,0.,0.)

        axis_list = []

        gs = GridSpec(ny, nx, figure=fig, hspace=0, wspace=0)

        for ichannel, channel in enumerate(channels):

            slc = self[channel,:,:]

            ax = plt.subplot(gs[ichannel], projection=slc.wcs)
            im = ax.imshow(slc.value, origin='lower', cmap=cmap, **kwargs)
            if contourkwargs:
                ax.contour(slc.value, **contourkwargs)
            ax.set_xlim(cenx - cenx / zoom, cenx + cenx / zoom)
            ax.set_ylim(ceny - ceny / zoom, ceny + ceny / zoom)

            if textcolor != 'notext':
                if textcolor is None:
                    # determine average image color and set textcolor to opposite
                    # (this is a bit hacky and there is _definitely_ a better way
                    # to do this)
                    avgcolor = im.cmap(im.norm(im.get_array())).mean(axis=(0,1))
                    totalcolor = avgcolor[:3].sum()
                    if totalcolor > 0.5:
                        textcolor = 'w'
                    else:
                        textcolor = 'k'

                ax.set_title(("{0:." + str(decimals) + "f}").format(spectral_axis[channel]),
                             x=textxloc, y=textyloc, color=textcolor)

            # only label bottom-left panel with locations
            if (ichannel != nx*(ny-1)):
                ax.coords[0].set_ticklabel_position('')
                ax.coords[1].set_ticklabel_position('')

            ax.tick_params(direction='in', color=textcolor)

            axis_list.append(ax)

        if tighten:
            plt.tight_layout()

        if output_file is not None:
            fig.savefig(output_file, **savefig_kwargs)

        return axis_list



class SpectralCube(BaseSpectralCube, BeamMixinClass):

    __name__ = "SpectralCube"

    _oned_spectrum = OneDSpectrum

    def __init__(self, data, wcs, mask=None, meta=None, fill_value=np.nan,
                 header=None, allow_huge_operations=False, beam=None,
                 wcs_tolerance=0.0, **kwargs):

        super(SpectralCube, self).__init__(data=data, wcs=wcs, mask=mask,
                                           meta=meta, fill_value=fill_value,
                                           header=header,
                                           allow_huge_operations=allow_huge_operations,
                                           wcs_tolerance=wcs_tolerance,
                                           **kwargs)

        # Beam loading must happen *after* WCS is read

        if beam is None:
            beam = cube_utils.try_load_beam(self.header)
        else:
            if not isinstance(beam, Beam):
                raise TypeError("beam must be a radio_beam.Beam object.")

        # Allow setting the beam attribute even if there is no beam defined
        # Accessing `SpectralCube.beam` without a beam defined raises a
        # `NoBeamError` with an informative message.
        self.beam = beam

        if beam is not None:
            self._meta['beam'] = beam
            self._header.update(beam.to_header_keywords())

    def _new_cube_with(self, **kwargs):
        beam = kwargs.pop('beam', None)
        if 'beam' in self._meta and beam is None:
            beam = self._beam
        newcube = super(SpectralCube, self)._new_cube_with(beam=beam, **kwargs)
        return newcube

    _new_cube_with.__doc__ = BaseSpectralCube._new_cube_with.__doc__

    def with_beam(self, beam):
        '''
        Attach a beam object to the `~SpectralCube`.

        Parameters
        ----------
        beam : `~radio_beam.Beam`
            `Beam` object defining the resolution element of the
            `~SpectralCube`.
        '''

        if not isinstance(beam, Beam):
            raise TypeError("beam must be a radio_beam.Beam object.")

        meta = self._meta.copy()
        meta['beam'] = beam

        header = self._header.copy()
        header.update(beam.to_header_keywords())

        newcube = self._new_cube_with(meta=self.meta, beam=beam)

        return newcube

class VaryingResolutionSpectralCube(BaseSpectralCube, MultiBeamMixinClass):
    """
    A variant of the SpectralCube class that has PSF (beam) information on a
    per-channel basis.
    """

    __name__ = "VaryingResolutionSpectralCube"

    _oned_spectrum = VaryingResolutionOneDSpectrum

    def __init__(self, *args, **kwargs):
        """
        Create a SpectralCube with an associated beam table.  The new
        VaryingResolutionSpectralCube will have a ``beams`` attribute and a
        ``beam_threshold`` attribute as described below.  It will perform some
        additional checks when trying to perform analysis across image frames.

        Three new keyword arguments are accepted:

        Other Parameters
        ----------------
        beam_table : `numpy.recarray`
            A table of beam major and minor axes in arcseconds and position
            angles, with labels BMAJ, BMIN, BPA
        beams : list
            A list of `radio_beam.Beam` objects
        beam_threshold : float or dict
            The fractional threshold above which beams are considered
            different.  A dictionary may be used with entries 'area', 'major',
            'minor', 'pa' so that you can specify a different fractional
            threshold for each of these.  For example, if you want to check
            only that the areas are the same, and not worry about the shape
            (which might be a bad idea...), you could set
            ``beam_threshold={'area':0.01, 'major':1.5, 'minor':1.5,
            'pa':5.0}``
        """
        # these types of cube are undefined without the radio_beam package

        beam_table = kwargs.pop('beam_table', None)
        beams = kwargs.pop('beams', None)
        beam_threshold = kwargs.pop('beam_threshold', 0.01)

        if (beam_table is None and beams is None):
            raise ValueError(
               "Must give either a beam table or a list of beams to "
               "initialize a VaryingResolutionSpectralCube")

        super(VaryingResolutionSpectralCube, self).__init__(*args, **kwargs)

        if isinstance(beam_table, BinTableHDU):
            beam_data_table = beam_table.data
        else:
            beam_data_table = beam_table

        if beam_table is not None:
            # CASA beam tables are in arcsec, and that's what we support
            beams = Beams(major=u.Quantity(beam_data_table['BMAJ'], u.arcsec),
                          minor=u.Quantity(beam_data_table['BMIN'], u.arcsec),
                          pa=u.Quantity(beam_data_table['BPA'], u.deg),
                          meta=[{key: row[key] for key in beam_data_table.names
                                 if key not in ('BMAJ','BPA', 'BMIN')}
                                for row in beam_data_table],
                         )
            goodbeams = beams.isfinite

            # track which, if any, beams are masked for later use
            self.goodbeams_mask = goodbeams

            if not all(goodbeams):
                warnings.warn("There were {0} non-finite beams; layers with "
                              "non-finite beams will be masked out.".format(
                                  np.count_nonzero(np.logical_not(goodbeams))),
                              NonFiniteBeamsWarning
                             )

            beam_mask = BooleanArrayMask(goodbeams[:,None,None],
                                         wcs=self._wcs,
                                         shape=self.shape,
                                        )
            if not is_broadcastable_and_smaller(beam_mask.shape,
                                                self._data.shape):
                # this should never be allowed to happen
                raise ValueError("Beam mask shape is not broadcastable to data shape: "
                                 "%s vs %s" % (beam_mask.shape, self._data.shape))
            assert beam_mask.shape == self.shape

            new_mask = np.bitwise_and(self._mask, beam_mask)

            new_mask._validate_wcs(new_data=self._data, new_wcs=self._wcs)

            self._mask = new_mask

        if (len(beams) != self.shape[0]):
            raise ValueError("Beam list must have same size as spectral "
                             "dimension")

        self.beams = beams
        self.beam_threshold = beam_threshold

    def __getitem__(self, view):

        # Need to allow self[:], self[:,:]
        if isinstance(view, (slice,int,np.int64)):
            view = (view, slice(None), slice(None))
        elif len(view) == 2:
            view = view + (slice(None),)
        elif len(view) > 3:
            raise IndexError("Too many indices")

        meta = {}
        meta.update(self._meta)
        slice_data = [(s.start, s.stop, s.step)
                      if hasattr(s,'start') else s
                      for s in view]
        if 'slice' in meta:
            meta['slice'].append(slice_data)
        else:
            meta['slice'] = [slice_data]

        # intslices identifies the slices that are given by integers, i.e.
        # indices.  Other slices are slice objects, e.g. obj[5:10], and have
        # 'start' attributes.
        intslices = [2-ii for ii,s in enumerate(view) if not hasattr(s,'start')]

        # for beams, we care only about the first slice, independent of its
        # type
        specslice = view[0]

        if intslices:
            if len(intslices) > 1:
                if 2 in intslices:
                    raise NotImplementedError("1D slices along non-spectral "
                                              "axes are not yet implemented.")
                newwcs = self._wcs.sub([a
                                        for a in (1,2,3)
                                        if a not in [x+1 for x in intslices]])
                if hasattr(self, '_beam') and self._beam is not None:
                    bmarg = {'beam': self.beam}
                elif hasattr(self, 'beams'):
                    bmarg = {'beams': self.unmasked_beams[specslice]}
                else:
                    bmarg = {}
                return self._oned_spectrum(value=self._data[view],
                                           wcs=newwcs,
                                           copy=False,
                                           unit=self.unit,
                                           spectral_unit=self._spectral_unit,
                                           mask=self.mask[view],
                                           meta=meta,
                                           goodbeams_mask=self.goodbeams_mask[specslice]
                                           if hasattr(self, '_goodbeams_mask')
                                           else None,
                                           **bmarg
                                          )

            # only one element, so drop an axis
            newwcs = wcs_utils.drop_axis(self._wcs, intslices[0])
            header = self._nowcs_header

            # Slice objects know how to parse Beam objects stored in the
            # metadata
            # A 2D slice with a VRSC should not be allowed along a
            # position-spectral axis
            if not isinstance(self.unmasked_beams[specslice], Beam):
                raise AttributeError("2D slices along a spectral axis are not "
                                     "allowed for "
                                     "VaryingResolutionSpectralCubes. Convolve"
                                     " to a common resolution with "
                                     "`convolve_to` before attempting "
                                     "position-spectral slicing.")

            meta['beam'] = self.unmasked_beams[specslice]
            return Slice(value=self.filled_data[view],
                         wcs=newwcs,
                         copy=False,
                         unit=self.unit,
                         header=header,
                         meta=meta)

        newmask = self._mask[view] if self._mask is not None else None

        newwcs = wcs_utils.slice_wcs(self._wcs, view, shape=self.shape)
        newwcs._naxis = list(self.shape)

        # this is an assertion to ensure that the WCS produced is valid
        # (this is basically a regression test for #442)
        assert newwcs[:, slice(None), slice(None)]
        assert len(newwcs._naxis) == 3

        return self._new_cube_with(data=self._data[view],
                                   wcs=newwcs,
                                   mask=newmask,
                                   beams=self.unmasked_beams[specslice],
                                   meta=meta)

    def spectral_slab(self, lo, hi):
        """
        Extract a new cube between two spectral coordinates

        Parameters
        ----------
        lo, hi : :class:`~astropy.units.Quantity`
            The lower and upper spectral coordinate for the slab range. The
            units should be compatible with the units of the spectral axis.
            If the spectral axis is in frequency-equivalent units and you
            want to select a range in velocity, or vice-versa, you should
            first use :meth:`~spectral_cube.SpectralCube.with_spectral_unit`
            to convert the units of the spectral axis.
        """

        # Find range of values for spectral axis
        ilo = self.closest_spectral_channel(lo)
        ihi = self.closest_spectral_channel(hi)

        if ilo > ihi:
            ilo, ihi = ihi, ilo
        ihi += 1

        # Create WCS slab
        wcs_slab = self._wcs.deepcopy()
        wcs_slab.wcs.crpix[2] -= ilo

        # Create mask slab
        if self._mask is None:
            mask_slab = None
        else:
            try:
                mask_slab = self._mask[ilo:ihi, :, :]
            except NotImplementedError:
                warnings.warn("Mask slicing not implemented for "
                              "{0} - dropping mask".
                              format(self._mask.__class__.__name__),
                              NotImplementedWarning
                             )
                mask_slab = None

        # Create new spectral cube
        slab = self._new_cube_with(data=self._data[ilo:ihi], wcs=wcs_slab,
                                   beams=self.unmasked_beams[ilo:ihi],
                                   mask=mask_slab)

        return slab

    def _new_cube_with(self, goodbeams_mask=None, **kwargs):
        beams = kwargs.pop('beams', self.unmasked_beams)
        beam_threshold = kwargs.pop('beam_threshold', self.beam_threshold)

        VRSC = VaryingResolutionSpectralCube
        newcube = super(VRSC, self)._new_cube_with(beams=beams,
                                                   beam_threshold=beam_threshold,
                                                   **kwargs)
        if goodbeams_mask is not None:
            newcube.goodbeams_mask = goodbeams_mask
            assert hasattr(newcube, '_goodbeams_mask')
        else:
            newcube.goodbeams_mask = np.isfinite(newcube.beams)
            assert hasattr(newcube, '_goodbeams_mask')

        return newcube

    _new_cube_with.__doc__ = BaseSpectralCube._new_cube_with.__doc__

    def _check_beam_areas(self, threshold, mean_beam, mask=None):
        """
        Check that the beam areas are the same to within some threshold
        """

        if mask is not None:
            assert len(mask) == len(self.unmasked_beams)
            mask = np.array(mask, dtype='bool')
        else:
            mask = np.ones(len(self.unmasked_beams), dtype='bool')

        qtys = dict(sr=self.unmasked_beams.sr,
                    major=self.unmasked_beams.major.to(u.deg),
                    minor=self.unmasked_beams.minor.to(u.deg),
                    # position angles are not really comparable
                    #pa=u.Quantity([bm.pa for bm in self.unmasked_beams], u.deg),
                   )

        errormessage = ""

        for (qtyname, qty) in (qtys.items()):
            minv = qty[mask].min()
            maxv = qty[mask].max()
            mn = getattr(mean_beam, qtyname)
            maxdiff = (np.max(np.abs(u.Quantity((maxv-mn, minv-mn))))/mn).decompose()

            if isinstance(threshold, dict):
                th = threshold[qtyname]
            else:
                th = threshold

            if maxdiff > th:
                errormessage += ("Beam {2}s differ by up to {0}x, which is greater"
                                 " than the threshold {1}\n".format(maxdiff,
                                                                    threshold,
                                                                    qtyname
                                                                   ))
        if errormessage != "":
            raise ValueError(errormessage)


    def __getattribute__(self, attrname):
        """
        For any functions that operate over the spectral axis, perform beam
        sameness checks before performing the operation to avoid unexpected
        results
        """

        # short name to avoid long lines below
        VRSC = VaryingResolutionSpectralCube

        # what about apply_numpy_function, apply_function?  since they're
        # called by some of these, maybe *only* those should be wrapped to
        # avoid redundant calls
        if attrname in ('moment', 'apply_numpy_function', 'apply_function',
                        'apply_function_parallel_spectral'):
            origfunc = super(VRSC, self).__getattribute__(attrname)
            return self._handle_beam_areas_wrapper(origfunc)
        else:
            return super(VRSC, self).__getattribute__(attrname)

    @property
    def hdu(self):
        raise ValueError("For VaryingResolutionSpectralCube's, use hdulist "
                         "instead of hdu.")

    @property
    def hdulist(self):
        """
        HDUList version of self
        """
        hdu = PrimaryHDU(self.filled_data[:].value, header=self.header)

        from .cube_utils import beams_to_bintable
        # use unmasked beams because, even if the beam is masked out, we should
        # write it
        bmhdu = beams_to_bintable(self.unmasked_beams)

        return HDUList([hdu, bmhdu])

    @warn_slow
    def convolve_to(self, beam, allow_smaller=False,
                    convolve=convolution.convolve_fft,
                    update_function=None):
        """
        Convolve each channel in the cube to a specified beam

        .. warning::
            The current implementation of ``convolve_to`` creates an in-memory
            copy of the whole cube to store the convolved data.  Issue #506
            notes that this is a problem, and it is on our to-do list to fix.

        .. warning::
            Note that if there is any misaligment between the cube's spatial
            pixel axes and the WCS's spatial axes *and* the beams are not
            round, the convolution kernels used here may be incorrect.  Be wary
            in such cases!

        Parameters
        ----------
        beam : `radio_beam.Beam`
            The beam to convolve to
        allow_smaller : bool
            If the specified target beam is smaller than the beam in a channel
            in any dimension and this is ``False``, it will raise an exception.
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`
        update_function : method
            Method that is called to update an external progressbar
            If provided, it disables the default `astropy.utils.console.ProgressBar`

        Returns
        -------
        cube : `SpectralCube`
            A SpectralCube with a single ``beam``
        """

        if ((self.wcs.celestial.wcs.get_pc()[0,1] != 0 or
             self.wcs.celestial.wcs.get_pc()[1,0] != 0)):
            warnings.warn("The beams will produce convolution kernels "
                          "that are not aware of any misaligment "
                          "between pixel and world coordinates, "
                          "and there are off-diagonal elements of the "
                          "WCS spatial transformation matrix.  "
                          "Unexpected results are likely.",
                          BeamWarning
                         )

        pixscale = wcs.utils.proj_plane_pixel_area(self.wcs.celestial)**0.5*u.deg

        convolution_kernels = []
        for bm,valid in zip(self.unmasked_beams, self.goodbeams_mask):
            if not valid:
                # just skip masked-out beams
                convolution_kernels.append(None)
                continue
            elif beam == bm:
                # Point response when beams are equal, don't convolve.
                convolution_kernels.append(None)
                continue
            try:
                cb = beam.deconvolve(bm)
                ck = cb.as_kernel(pixscale)
                convolution_kernels.append(ck)
            except ValueError:
                if allow_smaller:
                    convolution_kernels.append(None)
                else:
                    raise

        if update_function is None:
            pb = ProgressBar(self.shape[0])
            update_function = pb.update

        newdata = np.empty(self.shape)
        for ii,kernel in enumerate(convolution_kernels):

            # load each image from a slice to avoid loading whole cube into
            # memory
            img = self[ii,:,:].filled_data[:]

            # Kernel can only be None when `allow_smaller` is True,
            # or if the beams are equal. Only the latter is really valid.
            if kernel is None:
                newdata[ii, :, :] = img
            else:
                newdata[ii, :, :] = convolve(img, kernel,
                                             normalize_kernel=True)
            update_function()

        newcube = SpectralCube(data=newdata, wcs=self.wcs, mask=self.mask,
                               meta=self.meta, fill_value=self.fill_value,
                               header=self.header,
                               allow_huge_operations=self.allow_huge_operations,
                               beam=beam,
                               wcs_tolerance=self._wcs_tolerance)

        return newcube

    @warn_slow
    def to(self, unit, equivalencies=()):
        """
        Return the cube converted to the given unit (assuming it is equivalent).
        If conversion was required, this will be a copy, otherwise it will
        """

        if not isinstance(unit, u.Unit):
            unit = u.Unit(unit)

        if unit == self.unit:
            # No copying
            return self

        if self.unit.is_equivalent(u.Jy/u.beam):
            # replace "beam" with the actual beam
            if not hasattr(self, 'beams'):
                raise ValueError("To convert cubes with Jy/beam units, "
                                 "the cube needs to have beams defined.")
            factor = self.jtok_factors(equivalencies=equivalencies) * (self.unit*u.beam).to(u.Jy)
        else:
            # scaling factor
            factor = self.unit.to(unit, equivalencies=equivalencies)

        # special case: array in equivalencies
        # (I don't think this should have to be special cased, but I don't know
        # how to manipulate broadcasting rules any other way)
        if hasattr(factor, '__len__') and len(factor) == len(self):
            return self._new_cube_with(data=self._data*factor[:,None,None],
                                       unit=unit)
        else:
            return self._new_cube_with(data=self._data*factor,
                                       unit=unit)

    def mask_channels(self, goodchannels):
        """
        Helper function to mask out channels.  This function is equivalent to
        adding a mask with ``cube[view]`` where ``view`` is broadcastable to
        the cube shape, but it accepts 1D arrays that are not normally
        broadcastable.  Additionally, for `VaryingResolutionSpectralCube` s,
        the beams in the bad channels will not be checked when averaging,
        convolving, and doing other operations that are multibeam-aware.

        Parameters
        ----------
        goodchannels : array
            A 1D boolean array declaring which channels should be kept.

        Returns
        -------
        cube : `SpectralCube`
            A cube with the specified channels masked
        """
        goodchannels = np.asarray(goodchannels, dtype='bool')

        if goodchannels.ndim != 1:
            raise ValueError("goodchannels mask must be one-dimensional")
        if goodchannels.size != self.shape[0]:
            raise ValueError("goodchannels must have a length equal to the "
                             "cube's spectral dimension.")

        mask = BooleanArrayMask(goodchannels[:,None,None], self._wcs,
                                shape=self._data.shape)

        return self._new_cube_with(mask=mask,
                                   goodbeams_mask=np.logical_and(goodchannels,
                                                                 self.goodbeams_mask))



    def spectral_interpolate(self, *args, **kwargs):
        raise AttributeError("VaryingResolutionSpectralCubes can't be "
                             "spectrally interpolated.  Convolve to a "
                             "common resolution with `convolve_to` before "
                             "attempting spectral interpolation.")

    def spectral_smooth(self, *args, **kwargs):
        raise AttributeError("VaryingResolutionSpectralCubes can't be "
                             "spectrally smoothed.  Convolve to a "
                             "common resolution with `convolve_to` before "
                             "attempting spectral smoothed.")


def _regionlist_to_single_region(region_list):
    """
    Recursively merge a region list into a single compound region
    """
    import regions
    if len(region_list) == 1:
        return region_list[0]
    left = _regionlist_to_single_region(region_list[:len(region_list)//2])
    right = _regionlist_to_single_region(region_list[len(region_list)//2:])
    return regions.CompoundPixelRegion(left, right, operator.or_)
