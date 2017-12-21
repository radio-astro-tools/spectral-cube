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

import astropy.wcs
from astropy import units as u
from astropy.extern import six
from astropy.extern.six.moves import range as xrange
from astropy.extern.six.moves import zip
from astropy.io.fits import PrimaryHDU, BinTableHDU, Header, Card, HDUList
from astropy.utils.console import ProgressBar
from astropy import log
from astropy import wcs
from astropy import convolution
from astropy import stats

import numpy as np

from radio_beam import Beam, Beams

from . import cube_utils
from . import wcs_utils
from . import spectral_axis
from .masks import (LazyMask, LazyComparisonMask, BooleanArrayMask, MaskBase,
                    is_broadcastable_and_smaller)
from .ytcube import ytCube
from .lower_dimensional_structures import (Projection, Slice, OneDSpectrum,
                                           LowerDimensionalObject)
from .base_class import (BaseNDClass, SpectralAxisMixinClass,
                         DOPPLER_CONVENTIONS, SpatialCoordMixinClass,
                         MaskableArrayMixinClass)
from .utils import cached, warn_slow, VarianceWarning, BeamAverageWarning

from distutils.version import LooseVersion


__all__ = ['SpectralCube', 'VaryingResolutionSpectralCube']

# apply_everywhere, world: do not have a valid cube to test on
__doctest_skip__ = ['BaseSpectralCube._apply_everywhere']

try:
    from scipy import ndimage
    scipyOK = True
except ImportError:
    scipyOK = False

warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning, append=True)

SIGMA2FWHM = 2. * np.sqrt(2. * np.log(2.))


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

# convenience structures to keep track of the reversed index
# conventions between WCS and numpy
np2wcs = {2: 0, 1: 1, 0: 2}

class BaseSpectralCube(BaseNDClass, MaskableArrayMixinClass,
                       SpectralAxisMixinClass, SpatialCoordMixinClass):

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

        if spectral_axis.unit_from_header(self._header) is not None:
            self._spectral_unit = spectral_axis.unit_from_header(self._header)

        self._spectral_scale = spectral_axis.wcs_unit_scale(self._spectral_unit)

        self.allow_huge_operations = allow_huge_operations

        self._cache = {}

    @property
    def _is_huge(self):
        return cube_utils.is_huge(self)

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

    @property
    def unit(self):
        """ The flux unit """
        if self._unit:
            return self._unit
        else:
            return u.dimensionless_unscaled

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
        if self.unit is u.dimensionless_unscaled:
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
                                                    self.wcs.wcs.cunit[2],
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
        elif how not in ['auto', 'cube']:
            warnings.warn("Cannot use how=%s. Using how=cube" % how)

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
                    return OneDSpectrum(value=out,
                                        wcs=new_wcs,
                                        copy=False,
                                        unit=unit,
                                        header=header,
                                        meta=meta,
                                        spectral_unit=self._spectral_unit,
                                        beams=(self.beams
                                               if hasattr(self,'beams')
                                               else None),
                                       )
                else:
                    return out

            else:
                new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])
                header = self._nowcs_header

                return Projection(out, copy=False, wcs=new_wcs, meta=meta,
                                  unit=unit, header=header)
        else:
            return out


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
                    return OneDSpectrum(value=out,
                                        wcs=newwcs,
                                        copy=False,
                                        unit=self.unit,
                                        spectral_unit=self._spectral_unit,
                                        beams=(self.beams
                                               if hasattr(self, 'beams')
                                               else None),
                                        meta=self.meta)
                else:
                    raise NotImplementedError("We don't yet know how to deal "
                                              "with multidimensional averages "
                                              "that are non-spectral")
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
                raise NotImplementedError("Standard deviation across two "
                                          "dimensions slicewise is not "
                                          "yet implemented.")
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
    def mad_std(self, axis=None, **kwargs):
        """
        Use astropy's mad_std to computer the standard deviation
        """
        if int(astropy.__version__[0]) < 2:
            raise NotImplementedError("mad_std requires astropy >= 2")
        projection = self._naxes_dropped(axis) in (1,2)
        return self.apply_numpy_function(stats.mad_std, fill=np.nan,
                                         how='cube', axis=axis, unit=self.unit,
                                         ignore_nan=True,
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
            warnings.warn("Cube WCSs do not match, but their shapes do")
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
                       projection=False, progressbar=False, **kwargs):
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

        # determine the output array shape
        nx, ny = self._get_flat_shape(axis)

        # allocate memory for output array
        out = np.empty([nx, ny]) * np.nan

        if progressbar:
            progressbar = ProgressBar(nx*ny)
            pbu = progressbar.update
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
                    out[y, x] = result.value
                else:
                    out[y, x] = result
            pbu()

        if projection and axis in (0,1,2):
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

        for y in xrange(ny):
            for x in xrange(nx):
                # create length-1 view for each position
                slc = [slice(y, y + 1), slice(x, x + 1), ]
                # create a length-N slice (all-inclusive) along the selected axis
                slc.insert(axis, slice(None))
                yield y, x, slc

    def _iter_slices(self, axis, fill=np.nan, check_endian=False):
        """
        Iterate over the cube one slice at a time,
        replacing masked elements with fill
        """
        view = [slice(None)] * 3
        for x in range(self.shape[axis]):
            view[axis] = x
            yield self._get_filled_data(view=view, fill=fill,
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
                                     view=view,
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

    def flattened_world(self, view=()):
        """
        Retrieve the world coordinates corresponding to the extracted flattened
        version of the cube
        """

        # NOTE: this should be moved to SpatialCoordMixinClass once masks
        # are implemented for lower dim objects - EK

        lon,lat,spec = self.world[view]
        spec = self._mask._flattened(data=spec, wcs=self._wcs, view=slice)
        lon = self._mask._flattened(data=lon, wcs=self._wcs, view=slice)
        lat = self._mask._flattened(data=lat, wcs=self._wcs, view=slice)
        return lat,lon,spec

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
        if bnok and not iterate_rays:
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
        mask : :class:`MaskBase` instance, or boolean numpy array
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
            new_mask = self._mask & mask
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
                return OneDSpectrum(value=self._data[view],
                                    wcs=newwcs,
                                    copy=False,
                                    unit=self.unit,
                                    spectral_unit=self._spectral_unit,
                                    mask=self.mask[view],
                                    meta=meta,
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

    def with_fill_value(self, fill_value):
        """
        Create a new :class:`SpectralCube` with a different `fill_value`.

        Notes
        -----
        This method is fast (it does not copy any data)
        """
        return self._new_cube_with(fill_value=fill_value)

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
        return u.Quantity(self._data[view], self.unit, copy=False)

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
                              format(self._mask.__class__.__name__))
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
        region_mask: `masks.MaskBase` or boolean `numpy.ndarray`
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
        region_mask: `masks.MaskBase` or boolean `numpy.ndarray`
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

        return slices

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

        log.debug('slices: {0}'.format(slices))

        return self[slices]

    def subcube_from_ds9region(self, ds9region, allow_empty=False):
        """
        Extract a masked subcube from a ds9 region or a pyregion Region object
        (only functions on celestial dimensions)

        Parameters
        ----------
        ds9region: str or `pyregion.Shape`
            The region to extract
        allow_empty: bool
            If this is False, an exception will be raised if the region
            contains no overlap with the cube
        """
        import pyregion

        if isinstance(ds9region, six.string_types):
            shapelist = pyregion.parse(ds9region)
        else:
            shapelist = ds9region

        if shapelist[0].coord_format not in ('physical','image'):
            # Requires astropy >0.4...
            # pixel_regions = shapelist.as_imagecoord(self.wcs.celestial.to_header())
            # convert the regions to image (pixel) coordinates
            celhdr = self.wcs.sub([wcs.WCSSUB_CELESTIAL]).to_header()
            celhdr['NAXIS1'] = self.shape[2]
            celhdr['NAXIS2'] = self.shape[1]
            pixel_regions = shapelist.as_imagecoord(celhdr)
            recompute_shifted_mask = False
        else:
            pixel_regions = copy.deepcopy(shapelist)
            # we need to change the reference pixel after cropping
            recompute_shifted_mask = True

        # This is a hack to use mpl to determine the outer bounds of the regions
        # (but it's a legit hack - pyregion needs a major internal refactor
        # before we can approach this any other way, I think -AG)
        mpl_objs = pixel_regions.get_mpl_patches_texts(origin=0)[0]

        # Find the minimal enclosing box containing all of the regions
        # (this will speed up the mask creation below)
        extent = mpl_objs[0].get_extents()
        xlo, ylo = extent.min
        xhi, yhi = extent.max
        all_extents = [obj.get_extents() for obj in mpl_objs]
        for ext in all_extents:
            xlo = int(np.floor(xlo if xlo < ext.min[0] else ext.min[0]))
            ylo = int(np.floor(ylo if ylo < ext.min[1] else ext.min[1]))
            xhi = int(np.ceil(xhi if xhi > ext.max[0] else ext.max[0]))
            yhi = int(np.ceil(yhi if yhi > ext.max[1] else ext.max[1]))

        # Negative indices will do bad things, like wrap around the cube
        # If xhi/yhi are negative, there is not overlap
        if (xhi < 0) or (yhi < 0):
            raise ValueError("Region is outside of cube.")

        # if xlo/ylo are negative, we need to crop
        if xlo < 0:
            xlo = 0
        if ylo < 0:
            ylo = 0

        log.debug("Region boundaries: ")
        log.debug("xlo={xlo}, ylo={ylo}, xhi={xhi}, yhi={yhi}".format(xlo=xlo,
                                                                      ylo=ylo,
                                                                      xhi=xhi,
                                                                      yhi=yhi))

        subcube = self.subcube(xlo=xlo, ylo=ylo, xhi=xhi, yhi=yhi)

        if any(dim == 0 for dim in subcube.shape):
            if allow_empty:
                warnings.warn("The derived subset is empty: the region does not"
                              " overlap with the cube (but allow_empty=True).")
            else:
                raise ValueError("The derived subset is empty: the region does not"
                                 " overlap with the cube.")

        if recompute_shifted_mask:
            # for pixel-based regions (which we use in tests), we need to shift
            # the coordinates for mask computation because we're cropping the
            # cube
            for reg in pixel_regions:
                reg.params[0].v -= xlo
                reg.params[1].v -= ylo
                reg.params[0].text = str(reg.params[0].v)
                reg.params[1].text = str(reg.params[1].v)
                reg.coord_list[0] -= xlo
                reg.coord_list[1] -= ylo

            # use the pixel-based, shifted mask
            mask = pixel_regions.get_mask(header=subcube.wcs.celestial.to_header(),
                                          shape=subcube.shape[1:])
        else:
            # use the original, coordinate-based mask since the pixel mask has
            # *not* been shfited to match the original coordinate system
            celhdr = subcube.wcs.celestial.to_header()
            celhdr['NAXIS1'] = self.shape[2]
            celhdr['NAXIS2'] = self.shape[1]
            mask = shapelist.get_mask(header=celhdr,
                                      shape=subcube.shape[1:])

        if not allow_empty and mask.sum() == 0:
            raise ValueError("The derived subset is empty: the region does not"
                             " overlap with the cube.  However, this is likely "
                             "to be a bug, since at an earlier stage there was "
                             "overlap.")

        masked_subcube = subcube.with_mask(BooleanArrayMask(mask, subcube.wcs,
                                                            shape=subcube.shape))
        # by using ceil / floor above, we potentially introduced a NaN buffer
        # that we can now crop out
        return masked_subcube.minimal_subcube(spatial_only=True)

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


    @classmethod
    def read(cls, filename, format=None, hdu=None, **kwargs):
        """
        Read a spectral cube from a file.

        If the file contains Stokes axes, they will automatically be dropped.
        If you want to read in all Stokes informtion, use
        :meth:`~spectral_cube.StokesSpectralCube.read` instead.

        Parameters
        ----------
        filename : str
            The file to read the cube from
        format : str
            The format of the file to read. (Currently limited to 'fits' and 'casa_image')
        hdu : int or str
            For FITS files, the HDU to read in (can be the ID or name of an
            HDU).
        kwargs : dict
            If the format is 'fits', the kwargs are passed to
            :func:`~astropy.io.fits.open`.
        """
        from .io.core import read
        from .stokes_spectral_cube import StokesSpectralCube
        cube = read(filename, format=format, hdu=hdu, **kwargs)
        if isinstance(cube, StokesSpectralCube):
            if hasattr(cube, 'I'):
                warnings.warn("Cube is a Stokes cube, returning spectral cube for I component")
                return cube.I
            else:
                raise ValueError("Spectral cube is a Stokes cube that does not have an I component")
        else:
            return cube

    def write(self, filename, overwrite=False, format=None):
        """
        Write the spectral cube to a file.

        Parameters
        ----------
        filename : str
            The path to write the file to
        format : str
            The format of the file to write. (Currently limited to 'fits')
        overwrite : bool
            If True, overwrite ``filename`` if it exists
        """
        from .io.core import write
        write(filename, self, overwrite=overwrite, format=format)

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
        from glue.viewers.image.qt.viewer_widget import ImageWidget

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
                    self._glue_viewer = ga.new_data_viewer(ImageWidget,
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
        # Preserve non-WCS information from previous header iteration
        header = self._nowcs_header
        header.update(self.wcs.to_header())
        if self.unit == u.dimensionless_unscaled and 'BUNIT' in self._meta:
            # preserve the BUNIT even though it's not technically valid
            # (Jy/Beam)
            header['BUNIT'] = self._meta['BUNIT']
        else:
            header['BUNIT'] = self.unit.to_string(format='FITS')
        header.insert(2, Card(keyword='NAXIS', value=self._data.ndim))
        header.insert(3, Card(keyword='NAXIS1', value=self.shape[2]))
        header.insert(4, Card(keyword='NAXIS2', value=self.shape[1]))
        header.insert(5, Card(keyword='NAXIS3', value=self.shape[0]))

        # Preserve the cube's spectral units
        if self._spectral_unit != u.Unit(header['CUNIT3']):
            header['CDELT3'] *= self._spectral_scale
            header['CRVAL3'] *= self._spectral_scale
            header['CUNIT3'] = self._spectral_unit.to_string(format='FITS')

        if 'beam' in self._meta:
            header = self._meta['beam'].attach_to_header(header)

        # TODO: incorporate other relevant metadata here
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
            if not hasattr(self, 'beam'):
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
                      "https://github.com/radio-astro-tools/spectral-cube/issues")
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

    def reproject(self, header, order='bilinear'):
        """
        Reproject the cube into a new header.  Fills the data with the cube's
        ``fill_value`` to replace bad values before reprojection.

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
        """

        try:
            from reproject.version import version
        except ImportError:
            raise ImportError("Requires the reproject package to be"
                              " installed.")

        # Need version > 0.2 to work with cubes
        from distutils.version import LooseVersion
        if LooseVersion(version) < "0.3":
            raise Warning("Requires version >=0.3 of reproject. The current "
                          "version is: {}".format(version))

        from reproject import reproject_interp

        # TODO: Find the minimal subcube that contains the header and only reproject that
        # (see FITS_tools.regrid_cube for a guide on how to do this)

        newwcs = wcs.WCS(header)
        shape_out = [header['NAXIS{0}'.format(i + 1)] for i in range(header['NAXIS'])][::-1]

        newcube, newcube_valid = reproject_interp((self.filled_data[:],
                                                   self.header),
                                                  newwcs,
                                                  shape_out=shape_out,
                                                  order=order,
                                                  independent_celestial_slices=True)

        return self._new_cube_with(data=newcube,
                                   wcs=newwcs,
                                   mask=BooleanArrayMask(newcube_valid.astype('bool'),
                                                         newwcs),
                                   meta=self.meta,
                                  )


class SpectralCube(BaseSpectralCube):

    __name__ = "SpectralCube"

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

        if beam is not None:
            self.beam = beam
            self._meta['beam'] = beam
            self._header.update(beam.to_header_keywords())

            self.pixels_per_beam = (self.beam.sr /
                                    (astropy.wcs.utils.proj_plane_pixel_area(self.wcs) *
                                     u.deg**2)).to(u.dimensionless_unscaled).value
        self.abort = False  # Checked mid calculation
        self.external_update_function = None  # External function that updates a gui's progress bar

    def _new_cube_with(self, **kwargs):
        beam = kwargs.pop('beam', None)
        if 'beam' in self._meta and beam is None:
            beam = self.beam
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

    def spatial_smooth_median(self, ksize, **kwargs):
        """
        Smooth the image in each spatial-spatial plane of the cube using a median filter.

        Parameters
        ----------
        ksize : int
            Size of the median filter (scipy.ndimage.filters.median_filter)
        kwargs : dict
            Passed to the convolve function
        """
        if not scipyOK:
            raise ImportError("Scipy could not be imported: this function won't work.")

        shape = self.shape

        # "imagelist" is a generator
        # the boolean check will skip smoothing for bad spectra
        # TODO: should spatial good/bad be cached?
        imagelist = ((self.filled_data[ii],
                      self.mask.include(view=(ii, slice(None), slice(None))))
                      for ii in range(self.shape[0]))

        if self.external_update_function is None:
            pb = ProgressBar(shape[0])
            update_function = pb.update
        else:
            update_function = self.external_update_function

        def _gsmooth_image(args):
            """
            Helper function to smooth a spectrum
            """
            (im, includemask),kwargs = args
            if self.abort:
                return  # dump calculation
            else:
                update_function()

            if includemask.any():
                return ndimage.filters.median_filter(im, size=ksize)
            else:
                return im

        # could be numcores, except _gsmooth_spectrum is unpicklable
        with cube_utils._map_context(1) as map:
            smoothcube_ = np.array([x for x in
                                    map(_gsmooth_image, zip(imagelist,
                                                            itertools.cycle([kwargs]),
                                                           )
                                       )
                                  ])
        if self.abort:
            return

        # TODO: do something about the mask?
        newcube = self._new_cube_with(data=smoothcube_, wcs=self.wcs,
                                      mask=self.mask, meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube

    def spatial_smooth(self, kernel,
                       #numcores=None,
                       convolve=convolution.convolve, **kwargs):
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

        shape = self.shape

        # "imagelist" is a generator
        # the boolean check will skip smoothing for bad spectra
        # TODO: should spatial good/bad be cached?
        imagelist = ((self.filled_data[ii],
                     self.mask.include(view=(ii, slice(None), slice(None))))
                     for ii in range(self.shape[0]))

        if self.external_update_function is None:
            pb = ProgressBar(shape[0])
            update_function = pb.update
        else:
            update_function = self.external_update_function

        def _gsmooth_image(args):
            """
            Helper function to smooth an image
            """
            (im, includemask),kernel,kwargs = args
            if self.abort:
                return  # dump calculation
            else:
                update_function()

            if includemask.any():
                return convolve(im, kernel, normalize_kernel=True, **kwargs)
            else:
                return im

        # could be numcores, except _gsmooth_spectrum is unpicklable
        with cube_utils._map_context(1) as map:
            smoothcube_ = np.array([x for x in
                                    map(_gsmooth_image, zip(imagelist,
                                                            itertools.cycle([kernel]),
                                                            itertools.cycle([kwargs]),
                                                           )
                                       )
                                  ])
        if self.abort:
            return

        # TODO: do something about the mask?
        newcube = self._new_cube_with(data=smoothcube_, wcs=self.wcs,
                                      mask=self.mask, meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube

    def spectral_smooth_median(self, ksize, **kwargs):
        """
        Smooth the cube along the spectral dimension

        Parameters
        ----------
        ksize : int
            Size of the median filter (scipy.ndimage.filters.median_filter)
        kwargs : dict
            Passed to the convolve function
        """

        if not scipyOK:
            raise ImportError("Scipy could not be imported: this function won't work.")

        shape = self.shape

        # "cubelist" is a generator
        # the boolean check will skip smoothing for bad spectra
        # TODO: should spatial good/bad be cached?
        cubelist = ((self.filled_data[:,jj,ii],
                     self.mask.include(view=(slice(None), jj, ii)))
                    for jj in range(self.shape[1])
                    for ii in range(self.shape[2]))

        if self.external_update_function is None:
            pb = ProgressBar(shape[1] * shape[2])
            update_function = pb.update
        else:
            update_function = self.external_update_function

        def _gsmooth_spectrum(args):
            """
            Helper function to smooth a spectrum
            """
            (spec, includemask),kwargs = args
            if self.abort:
                return  # dump calculation
            else:
                update_function()

            if any(includemask):
                return ndimage.filters.median_filter(spec, size=ksize)
            else:
                return spec

        # could be numcores, except _gsmooth_spectrum is unpicklable
        with cube_utils._map_context(1) as map:
            smoothcube_ = np.array([x for x in
                                    map(_gsmooth_spectrum, zip(cubelist,
                                                               itertools.cycle([kwargs]),
                                                              )
                                       )
                                   ]
                                  )
        if self.abort:
            return

        # empirical: need to swapaxes to get shape right
        # cube = np.arange(6*5*4).reshape([4,5,6]).swapaxes(0,2)
        # cubelist.T.reshape(cube.shape) == cube
        smoothcube = smoothcube_.T.reshape(shape)

        # TODO: do something about the mask?
        newcube = self._new_cube_with(data=smoothcube, wcs=self.wcs,
                                      mask=self.mask, meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube

    def spectral_smooth(self, kernel,
                        #numcores=None,
                        convolve=convolution.convolve,
                        **kwargs):
        """
        Smooth the cube along the spectral dimension

        Parameters
        ----------
        kernel : `~astropy.convolution.Kernel1D`
            A 1D kernel from astropy
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`
        kwargs : dict
            Passed to the convolve function
        """

        shape = self.shape

        # "cubelist" is a generator
        # the boolean check will skip smoothing for bad spectra
        # TODO: should spatial good/bad be cached?
        cubelist = ((self.filled_data[:,jj,ii],
                     self.mask.include(view=(slice(None), jj, ii)))
                    for jj in range(self.shape[1])
                    for ii in range(self.shape[2]))

        if self.external_update_function is None:
            pb = ProgressBar(shape[1] * shape[2])
            update_function = pb.update
        else:
            update_function = self.external_update_function

        def _gsmooth_spectrum(args):
            """
            Helper function to smooth a spectrum
            """
            (spec, includemask),kernel,kwargs = args
            if self.abort:
                return  # dump calculation
            else:
                update_function()

            if any(includemask):
                return convolve(spec, kernel, normalize_kernel=True, **kwargs)
            else:
                return spec

        # could be numcores, except _gsmooth_spectrum is unpicklable
        with cube_utils._map_context(1) as map:
            smoothcube_ = np.array([x for x in
                                    map(_gsmooth_spectrum, zip(cubelist,
                                                               itertools.cycle([kernel]),
                                                               itertools.cycle([kwargs]),
                                                              )
                                       )
                                   ]
                                  )
        if self.abort:
            return

        # empirical: need to swapaxes to get shape right
        # cube = np.arange(6*5*4).reshape([4,5,6]).swapaxes(0,2)
        # cubelist.T.reshape(cube.shape) == cube
        smoothcube = smoothcube_.T.reshape(shape)

        # TODO: do something about the mask?
        newcube = self._new_cube_with(data=smoothcube, wcs=self.wcs,
                                      mask=self.mask, meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube

    def spectral_interpolate(self, spectral_grid,
                             suppress_smooth_warning=False,
                             fill_value=None):
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
                          "be smoothed prior to resampling.")

        newcube = np.empty([spectral_grid.size, self.shape[1], self.shape[2]],
                           dtype=cubedata[:1, 0, 0].dtype)
        newmask = np.empty([spectral_grid.size, self.shape[1], self.shape[2]],
                           dtype='bool')

        yy,xx = np.indices(self.shape[1:])

        pb = ProgressBar(xx.size)
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

            pb.update()

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

    def convolve_to(self, beam, convolve=convolution.convolve_fft):
        """
        Convolve each channel in the cube to a specified beam

        Parameters
        ----------
        beam : `radio_beam.Beam`
            The beam to convolve to
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`

        Returns
        -------
        cube : `SpectralCube`
            A SpectralCube with a single ``beam``
        """

        pixscale = wcs.utils.proj_plane_pixel_area(self.wcs.celestial)**0.5*u.deg

        convolution_kernel = beam.deconvolve(self.beam).as_kernel(pixscale)

        pb = ProgressBar(self.shape[0])

        newdata = np.empty(self.shape)
        for ii,img in enumerate(self.filled_data[:]):
            newdata[ii,:,:] = convolve(img, convolution_kernel,
                                       normalize_kernel=True)
            pb.update()

        newcube = self._new_cube_with(data=newdata, beam=beam)

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

    def abort_function(self):
        """
        Toggles abort value to true.
        Used interrupt smoothing calculations.
        """
        self.abort = True


class VaryingResolutionSpectralCube(BaseSpectralCube):
    """
    A variant of the SpectralCube class that has PSF (beam) information on a
    per-channel basis.
    """

    __name__ = "VaryingResolutionSpectralCube"

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
                          meta=[{key: row[key] for key in beam_table.names
                                 if key not in ('BMAJ','BPA', 'BMIN')}
                                for row in beam_data_table],
                         )
            goodbeams = beams.isfinite

            # track which, if any, beams are masked for later use
            self._goodbeams_mask = goodbeams

            if not all(goodbeams):
                warnings.warn("There were {0} non-finite beams; layers with "
                              "non-finite beams will be masked out.".format(
                                  np.count_nonzero(~goodbeams)))

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

            new_mask = self._mask & beam_mask

            new_mask._validate_wcs(new_data=self._data, new_wcs=self._wcs)

            self._mask = new_mask

        if (len(beams) != self.shape[0]):
            raise ValueError("Beam list must have same size as spectral "
                             "dimension")

        self._beams = beams
        self.beam_threshold = beam_threshold

    @property
    def beams(self):
        return self._beams

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
                return OneDSpectrum(value=self._data[view],
                                    wcs=newwcs,
                                    copy=False,
                                    unit=self.unit,
                                    spectral_unit=self._spectral_unit,
                                    mask=self.mask[view],
                                    beams=self.beams[specslice],
                                    meta=meta)

            # only one element, so drop an axis
            newwcs = wcs_utils.drop_axis(self._wcs, intslices[0])
            header = self._nowcs_header

            # Slice objects know how to parse Beam objects stored in the
            # metadata
            # A 2D slice with a VRSC should not be allowed along a
            # position-spectral axis
            if not isinstance(self.beams[specslice], Beam):
                raise AttributeError("2D slices along a spectral axis are not "
                                     "allowed for "
                                     "VaryingResolutionSpectralCubes. Convolve"
                                     " to a common resolution with "
                                     "`convolve_to` before attempting "
                                     "position-spectral slicing.")

            meta['beam'] = self.beams[specslice]
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

        return self._new_cube_with(data=self._data[view],
                                   wcs=newwcs,
                                   mask=newmask,
                                   beams=self.beams[specslice],
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
                              format(self._mask.__class__.__name__))
                mask_slab = None

        # Create new spectral cube
        slab = self._new_cube_with(data=self._data[ilo:ihi], wcs=wcs_slab,
                                   beams=self.beams[ilo:ihi],
                                   mask=mask_slab)

        return slab

    def _new_cube_with(self, goodbeams_mask=None, **kwargs):
        beams = kwargs.pop('beams', self.beams)
        beam_threshold = kwargs.pop('beam_threshold', self.beam_threshold)

        VRSC = VaryingResolutionSpectralCube
        newcube = super(VRSC, self)._new_cube_with(beams=beams,
                                                   beam_threshold=beam_threshold,
                                                   **kwargs)
        if goodbeams_mask is not None:
            newcube._goodbeams_mask = goodbeams_mask
        else:
            newcube._goodbeams_mask = newcube.beams.isfinite

        return newcube

    _new_cube_with.__doc__ = BaseSpectralCube._new_cube_with.__doc__

    def _check_beam_areas(self, threshold, mean_beam, mask=None):
        """
        Check that the beam areas are the same to within some threshold
        """

        if mask is not None:
            assert len(mask) == len(self.beams)
            mask = np.array(mask, dtype='bool')
        else:
            mask = np.ones(len(self.beams), dtype='bool')

        qtys = dict(sr=self.beams.sr,
                    major=self.beams.major.to(u.deg),
                    minor=self.beams.minor.to(u.deg),
                    # position angles are not really comparable
                    #pa=u.Quantity([bm.pa for bm in self.beams], u.deg),
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

    def identify_bad_beams(self, threshold, reference_beam=None,
                           criteria=['sr','major','minor'],
                           mid_value=np.nanmedian):
        """
        Mask out any layers in the cube that have beams that differ from the
        central value of the beam by more than the specified threshold.
        An acceptable beam area can also be specified directly.

        Parameters
        ----------
        threshold : float
            Fractional threshold
        reference_beam : Beam
            A beam to use as the reference.  If unspecified, ``mid_value`` will
            be used to select a middle beam
        criteria : list
            A list of criteria to compare.  Can include
            'sr','major','minor','pa' or any subset of those.
        mid_value : function
            The function used to determine the 'mid' value to compare to.  This
            will identify the middle-valued beam area.

        Returns
        -------
        includemask : np.array
            A boolean array where ``True`` indicates the good beams
        """

        includemask = np.ones(self.beams.size, dtype='bool')

        all_criteria = ['sr','major','minor','pa']
        if not set.issubset(set(criteria), set(all_criteria)):
            raise ValueError("Criteria must be one of the allowed options: "
                             "{0}".format(all_criteria))

        props = {prop: u.Quantity([getattr(beam, prop) for beam in self.beams])
                 for prop in all_criteria}

        if reference_beam is None:
            reference_beam = Beam(major=mid_value(props['major']),
                                  minor=mid_value(props['minor']),
                                  pa=mid_value(props['pa'])
                                 )

        for prop in criteria:
            val = props[prop]
            mid = getattr(reference_beam, prop)

            diff = np.abs((val-mid)/mid)

            assert diff.shape == includemask.shape

            includemask[diff > threshold] = False

        return includemask

    def mask_out_bad_beams(self, threshold, reference_beam=None,
                           criteria=['sr','major','minor'],
                           mid_value=np.nanmedian):
        """
        See `identify_bad_beams`.  This function returns a masked cube

        Returns
        -------
        newcube : VaryingResolutionSpectralCube
            The cube with bad beams masked out
        """

        goodbeams = self.identify_bad_beams(threshold=threshold,
                                            reference_beam=reference_beam,
                                            criteria=criteria,
                                            mid_value=mid_value)

        includemask = BooleanArrayMask(goodbeams[:,None,None],
                                       self._wcs,
                                       shape=self._data.shape)

        return self._new_cube_with(mask=self.mask & includemask,
                                   beam_threshold=threshold,
                                   goodbeams_mask=self._goodbeams_mask & goodbeams,
                                  )


    def average_beams(self, threshold, mask='compute', warn=False):
        """
        Average the beams.  Note that this operation only makes sense in
        limited contexts!  Generally one would want to convolve all the beams
        to a common shape, but this method is meant to handle the "simple" case
        when all your beams are the same to within some small factor and can
        therefore be arithmetically averaged.

        Parameters
        ----------
        threshold : float
            The fractional difference between beam major, minor, and pa to
            permit
        mask : 'compute', None, or boolean array
            The mask to apply to the beams.  Useful for excluding bad channels
            and edge beams.
        warn : bool
            Warn if successful?

        Returns
        -------
        new_beam : radio_beam.Beam
            A new radio beam object that is the average of the unmasked beams
        """
        if mask == 'compute':
            beam_mask = np.any(self.mask.include() &
                               self._goodbeams_mask[:,None,None],
                               axis=(1,2))
        else:
            if mask.ndim > 1:
                beam_mask = mask & self._goodbeams_mask[:,None,None]
            else:
                beam_mask = mask & self._goodbeams_mask

        new_beam = self.beams.average_beam(includemask=beam_mask)

        if np.isnan(new_beam):
            raise ValueError("Beam was not finite after averaging.  "
                             "This either indicates that there was a problem "
                             "with the include mask, one of the beam's values, "
                             "or a bug.")

        self._check_beam_areas(threshold, mean_beam=new_beam, mask=beam_mask)
        if warn:
            warnings.warn("Arithmetic beam averaging is being performed.  This is "
                          "not a mathematically robust operation, but is being "
                          "permitted because the beams differ by "
                          "<{0}".format(threshold),
                          BeamAverageWarning
                         )
        return new_beam


    def _handle_beam_areas_wrapper(self, function, beam_threshold=None):
        """
        Wrapper: if the function takes "axis" and is operating over axis 0 (the
        spectral axis), check that the beam threshold is not exceeded before
        performing the operation

        Also, if the operation *is* valid, average the beam appropriately to
        get the output
        """

        if beam_threshold is None:
            beam_threshold = self.beam_threshold

        def newfunc(*args, **kwargs):
            """ Wrapper function around the standard operations to handle beams
            when creating projections """

            # check that the spectral axis is being operated over.  If it is,
            # we need to average beams
            # moments are a special case b/c they default to axis=0
            need_to_handle_beams = (('axis' in kwargs and
                                     ((kwargs['axis']==0) or
                                      (hasattr(kwargs['axis'], '__len__') and
                                       0 in kwargs['axis']))) or
                                    ('axis' not in kwargs and 'moment' in
                                     function.__name__))

            if need_to_handle_beams:
                # do this check *first* so we don't do an expensive operation
                # and crash afterward
                avg_beam = self.average_beams(beam_threshold, warn=True)

            result = function(*args, **kwargs)

            if not isinstance(result, LowerDimensionalObject):
                # numpy arrays are sometimes returned; these have no metadata
                return result

            elif need_to_handle_beams:
                result.meta['beam'] = avg_beam
                result._beam = avg_beam

            return result

        return newfunc

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
        if attrname in ('moment', 'apply_numpy_function', 'apply_function'):
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
        bmhdu = beams_to_bintable(self.beams)

        return HDUList([hdu, bmhdu])

    def convolve_to(self, beam, allow_smaller=False,
                    convolve=convolution.convolve_fft):
        """
        Convolve each channel in the cube to a specified beam

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
                          "Unexpected results are likely.")

        pixscale = wcs.utils.proj_plane_pixel_area(self.wcs.celestial)**0.5*u.deg

        convolution_kernels = []
        for bm,valid in zip(self.beams, self._goodbeams_mask):
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

        pb = ProgressBar(self.shape[0])

        newdata = np.empty(self.shape)
        for ii,(img,kernel) in enumerate(zip(self.filled_data[:],
                                             convolution_kernels)):
            # Kernel can only be None when `allow_smaller` is True,
            # or if the beams are equal. Only the latter is really valid.
            if kernel is None:
                newdata[ii, :, :] = img
            else:
                newdata[ii, :, :] = convolve(img, kernel,
                                             normalize_kernel=True)
            pb.update()

        newcube = SpectralCube(data=newdata, wcs=self.wcs, mask=self.mask,
                               meta=self.meta, fill_value=self.fill_value,
                               header=self.header,
                               allow_huge_operations=self.allow_huge_operations,
                               beam=beam,
                               wcs_tolerance=self._wcs_tolerance)

        return newcube

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
    
    def jtok_factors(self, equivalencies=()):
        """
        Compute an array of multiplicative factors that will convert from
        Jy/beam to K
        """

        factors = []
        for bm,frq in zip(self.beams,
                          self.with_spectral_unit(u.Hz).spectral_axis):

            # create a beam equivalency for brightness temperature
            bmequiv = bm.jtok_equiv(frq)
            factor = (u.Jy).to(u.K, equivalencies=bmequiv+list(equivalencies))
            factors.append(factor)
        factor = np.array(factors)

        return factor

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
                                   goodbeams_mask=goodchannels & self._goodbeams_mask)
