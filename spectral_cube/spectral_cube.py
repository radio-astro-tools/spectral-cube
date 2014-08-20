"""
A class to represent a 3-d position-position-velocity spectral cube.
"""

import warnings
from functools import wraps

from astropy import units as u
from astropy.extern import six
from astropy.io.fits import PrimaryHDU, ImageHDU

import numpy as np

from . import cube_utils
from . import wcs_utils
from . import spectral_axis
from .masks import LazyMask, BooleanArrayMask
from .io.core import determine_format

__all__ = ['SpectralCube']

try:  # TODO replace with six.py
    xrange
except NameError:
    xrange = range


DOPPLER_CONVENTIONS = {}
DOPPLER_CONVENTIONS['radio'] = u.doppler_radio
DOPPLER_CONVENTIONS['optical'] = u.doppler_optical
DOPPLER_CONVENTIONS['relativistic'] = u.doppler_relativistic


def cached(func):
    """
    Decorator to cache function calls
    """
    cache = {}

    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper

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


class Projection(u.Quantity):

    def __new__(cls, value, unit=None, dtype=None, copy=True, wcs=None, meta=None):

        if value.ndim != 2:
            raise ValueError("value should be a 2-d array")

        if wcs is not None and wcs.wcs.naxis != 2:
            raise ValueError("wcs should have two dimension")

        self = u.Quantity.__new__(cls, value, unit=unit, dtype=dtype, copy=copy).view(cls)
        self._wcs = wcs
        self._meta = meta

        return self

    @property
    def wcs(self):
        return self._wcs

    @property
    def meta(self):
        return self._meta

    @property
    def hdu(self):
        from astropy.io import fits
        if self.wcs is None:
            hdu = fits.PrimaryHDU(self.value)
        else:
            hdu = fits.PrimaryHDU(self.value, header=self.wcs.to_header())
        hdu.header['BUNIT'] = self.unit.to_string(format='fits')
        return hdu

    def write(self, filename, format=None, overwrite=False):
        """
        Write the moment map to a file.

        Parameters
        ----------
        filename : str
            The path to write the file to
        format : str
            The kind of file to write. (Currently limited to 'fits')
        overwrite : bool
            If True, overwrite `filename` if it exists
        """
        if format is None:
            format = determine_format(filename)
        if format == 'fits':
            self.hdu.writeto(filename, clobber=overwrite)
        else:
            raise ValueError("Unknown format '{0}' - the only available "
                             "format at this time is 'fits'")

# A slice is just like a projection in every way
class Slice(Projection):
    pass


class SpectralCube(object):

    def __init__(self, data, wcs, mask=None, meta=None, fill_value=np.nan):

        # Deal with metadata first because it can affect data reading
        self._meta = meta or {}
        if 'BUNIT' in self._meta:
            try:
                self._unit = u.Unit(self._meta['BUNIT'])
            except ValueError:
                warnings.warn("Could not parse unit {0}".format(self._meta['BUNIT']))
                self._unit = None
        elif hasattr(data, 'unit'):
            self._unit = data.unit
            # strip the unit so that it can be treated as cube metadata
            data = data.value
        else:
            self._unit = None

        # TODO: mask should be oriented? Or should we assume correctly oriented here?
        self._data, self._wcs = cube_utils._orient(data, wcs)
        self._spectral_axis = None
        self._mask = mask  # specifies which elements to Nan/blank/ignore
                           # object or array-like object, given that WCS needs to be consistent with data?
        #assert mask._wcs == self._wcs
        self._fill_value = fill_value

        # We don't pass the spectral unit via the initializer since the user
        # should be using ``with_spectral_unit`` if they want to set it.
        # However, we do want to keep track of what units the spectral axis
        # should be returned in, otherwise astropy's WCS can change the units,
        # e.g. km/s -> m/s.
        self._spectral_unit = u.Unit(self._wcs.wcs.cunit[2])
        self._spectral_scale = 1.0

    def _new_cube_with(self, data=None, wcs=None, mask=None, meta=None,
                       fill_value=None, spectral_unit=None):

        data = self._data if data is None else data
        wcs = self._wcs if wcs is None else wcs
        mask = self._mask if mask is None else mask
        meta = self._meta if meta is None else meta
        fill_value = self._fill_value if fill_value is None else fill_value
        spectral_unit = self._spectral_unit if spectral_unit is None else spectral_unit

        cube = SpectralCube(data=data, wcs=wcs, mask=mask, meta=meta,
                            fill_value=fill_value)
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
    def ndim(self):
        """ Dimensionality of the data """
        return self._data.ndim

    def __repr__(self):
        s = "SpectralCube with shape={0}".format(self.shape)
        if self.unit is u.dimensionless_unscaled:
            s += ":\n"
        else:
            s += " and unit={0}:\n".format(self.unit)
        s += " n_x: {0}  type_x: {1:8s}  unit_x: {2}\n".format(self.shape[2], self.wcs.wcs.ctype[0], self.wcs.wcs.cunit[0])
        s += " n_y: {0}  type_y: {1:8s}  unit_y: {2}\n".format(self.shape[1], self.wcs.wcs.ctype[1], self.wcs.wcs.cunit[1])
        s += " n_s: {0}  type_s: {1:8s}  unit_s: {2}".format(self.shape[0], self.wcs.wcs.ctype[2], self.wcs.wcs.cunit[2])
        return s

    def apply_numpy_function(self, function, fill=np.nan,
                             reduce=True, how='auto',
                             projection=False,
                             unit=None,
                             check_endian=False, **kwargs):
        """
        Apply a numpy function to the cube

        Parameters
        ----------
        function : `numpy.ufunc`
            A numpy ufunc to apply to the cube
        fill : float
            The fill value to use on the data
        reduce : bool
            reduce indicates whether this is a reduce-like operation,
            that can be accumulated one slice at a time.
            sum/max/min are like this. argmax/argmin are not
        how : cube | slice | ray | auto
           How to compute the moment. All strategies give the same
           result, but certain strategies are more efficient depending
           on data size and layout. Cube/slice/ray iterate over
           decreasing subsets of the data, to conserve memory.
           Default='auto'
        projection : bool
            Return a projection if the resulting array is 2D?
        unit : None or `astropy.units.Unit`
            The unit to include for the output array.  For example,
            `SpectralCube.max` calls `SpectralCube.apply_numpy_function(np.max,
            unit=self.unit)`, inheriting the unit from the original cube.
            However, for other numpy functions, e.g. `numpy.argmax`, the return
            is an index and therefore unitless.
        check_endian : bool
            A flag to check the endianness of the data before applying the
            function.  This is only needed for optimized functions, e.g. those
            in the `bottleneck` package.
        kwargs : dict
            Passed to the numpy function.

        Returns
        -------
        result : `Projection` or `~astropy.units.Quantity` or float
            The result depends on the value of ``axis``, ``projection``, and
            ``unit``.  If ``axis`` is None, the return will be a scalar with or
            without units.  If axis is an integer, the return will be a
            `Projection` if ``projection`` is set
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

        if strategy == 'slice' and reduce:
            try:
                out = self._reduce_slicewise(function, fill,
                                              check_endian,
                                              **kwargs)
            except NotImplementedError:
                pass

        if how not in ['auto', 'cube']:
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
            new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])

            meta = {'collapse_axis': axis}

            return Projection(out, copy=False, wcs=new_wcs, meta=meta, unit=unit)
        else:
            return out


    def _reduce_slicewise(self, function, fill, check_endian, **kwargs):
        """
        Compute a numpy aggregation by grabbing one slice at a time
        """

        ax = kwargs.pop('axis', None)
        full_reduce = ax is None
        ax = ax or 0

        if isinstance(ax, tuple):
            raise NotImplementedError("Multi-axis reductions are not "
                                      "supported with how='slice'")

        planes = self._iter_slices(ax, fill=fill, check_endian=check_endian)
        result = next(planes)
        for plane in planes:
            result = function(np.dstack((result, plane)), axis=2, **kwargs)

        if full_reduce:
            result = function(result)

        return result

    def get_mask_array(self):
        """
        Convert the mask to a boolean numpy array
        """
        return self._mask.include(data=self._data, wcs=self._wcs)

    @property
    def mask(self):
        """
        The underlying mask
        """
        return self._mask

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
    def sum(self, axis=None, how='auto'):
        """
        Return the sum of the cube, optionally over an axis.
        """

        projection = self._naxes_dropped(axis) == 1

        # use nansum, and multiply by mask to add zero each time there is badness
        return self.apply_numpy_function(np.nansum, fill=np.nan, how=how,
                                         axis=axis, unit=self.unit,
                                         projection=projection)

    @aggregation_docstring
    def max(self, axis=None, how='auto'):
        """
        Return the maximum data value of the cube, optionally over an axis.
        """

        projection = self._naxes_dropped(axis) == 1

        return self.apply_numpy_function(np.nanmax, fill=np.nan, how=how,
                                         axis=axis, unit=self.unit,
                                         projection=projection)

    @aggregation_docstring
    def min(self, axis=None, how='auto'):
        """
        Return the minimum data value of the cube, optionally over an axis.
        """

        projection = self._naxes_dropped(axis) == 1

        return self.apply_numpy_function(np.nanmin, fill=np.nan, how=how,
                                         axis=axis, unit=self.unit,
                                         projection=projection)

    @aggregation_docstring
    def argmax(self, axis=None, how='auto'):
        """
        Return the index of the maximum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return self.apply_numpy_function(np.nanargmax, fill=-np.inf,
                                         reduce=False, projection=False,
                                         how=how, axis=axis)

    @aggregation_docstring
    def argmin(self, axis=None, how='auto'):
        """
        Return the index of the minimum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask
        """
        return self.apply_numpy_function(np.nanargmin, fill=np.inf,
                                         reduce=False, projection=False,
                                         how=how, axis=axis)

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

    def apply_function(self, function, axis=None, weights=None, unit=None,
                       projection=False, **kwargs):
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

        Returns
        -------
        result : `Projection` or `~astropy.units.Quantity` or float
            The result depends on the value of ``axis``, ``projection``, and
            ``unit``.  If ``axis`` is None, the return will be a scalar with or
            without units.  If axis is an integer, the return will be a
            `Projection` if ``projection`` is set
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

        # iterate over "lines of sight" through the cube
        for x, y, slc in self._iter_rays(axis):
            # acquire the flattened, valid data for the slice
            data = self.flattened(slc, weights=weights)
            if len(data) != 0:
                # store result in array
                out[x, y] = function(data, **kwargs)

        if projection and axis in (0,1,2):
            new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])

            meta = {'collapse_axis': axis}

            return Projection(out, copy=False, wcs=new_wcs, meta=meta, unit=unit)
        else:
            return out

    def _iter_rays(self, axis=None):
        """
        Iterate over view corresponding to lines-of-sight through a cube
        along the specified axis
        """
        nx, ny = self._get_flat_shape(axis)

        for x in xrange(nx):
            for y in xrange(ny):
                # create length-1 view for each position
                slc = [slice(x, x + 1), slice(y, y + 1)]
                # create a length-N slice (all-inclusive) along the selected axis
                slc.insert(axis, slice(None))
                yield x, y, slc

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

    def median(self, axis=None, **kwargs):
        """
        Compute the median of an array, optionally along an axis.

        Ignores excluded mask elements.

        Parameters
        ----------
        axis : int (optional)
            The axis to collapse

        Returns
        -------
        med : ndarray
            The median
        """
        try:
            from bottleneck import nanmedian
            result = self.apply_numpy_function(nanmedian, axis=axis,
                                                projection=True,
                                                check_endian=True, **kwargs)
        except ImportError:
            return self.apply_function(np.median, axis=axis, **kwargs)

    def percentile(self, q, axis=None, **kwargs):
        """
        Return percentiles of the data.

        Parameters
        ----------
        q : float
            The percentile to compute
        axis : int, or None
            Which axis to compute percentiles over
        """
        return self.apply_function(np.percentile, q=q, axis=axis, **kwargs)

    def with_mask(self, mask, inherit_mask=True):
        """
        Return a new SpectralCube instance that contains a composite mask of
        the current SpectralCube and the new ``mask``.

        Parameters
        ----------
        mask : :class:`MaskBase` instance, or boolean numpy array
            The mask to apply. If a boolean array is supplied,
            it will be converted into a mask, assuming that
            True values indicate included elements.

        inherit_mask : bool (optional, default=True)
            If True, combines the provided mask with the
            mask currently attached to the cube

        Returns
        -------
        new_cube : :class:`SpectralCube`
            A cube with the new mask applied.

        Notes
        -----
        This operation returns a view into the data, and not a copy.
        """
        if isinstance(mask, np.ndarray):
            if mask.shape != self._data.shape:
                raise ValueError("Mask shape doesn't match data shape: "
                                 "%s vs %s" % (mask.shape, self._data.shape))
            mask = BooleanArrayMask(mask, self._wcs)

        return self._new_cube_with(mask=self._mask & mask if inherit_mask else mask)

    def __getitem__(self, view):
        meta = {}
        meta.update(self._meta)
        meta['slice'] = [(s.start, s.stop, s.step)
                         if hasattr(s,'start') else s
                         for s in view]

        intslices = [ii for ii,s in enumerate(view) if not hasattr(s,'start')]

        if intslices:
            if len(intslices) > 1:
                raise NotImplementedError("1D slices are not implemented yet.")
            newwcs = self._wcs.dropaxis(intslices[0])
            newwcs = wcs_utils.slice_wcs(newwcs, [s for s in view
                                                  if hasattr(s,'start')])
            return Slice(value=self._data[view],
                         wcs=newwcs,
                         copy=False,
                         #mask=self._mask[view],
                         meta=meta)

        newmask = self._mask[view] if self._mask is not None else None

        return self._new_cube_with(data=self._data[view],
                                   wcs=wcs_utils.slice_wcs(self._wcs, view),
                                   mask=newmask,
                                   meta=meta)

    @property
    def fill_value(self):
        """ The replacement value used by :meth:`filled_data`.

        fill_value is immutable; use :meth:`with_fill_value`
        to create a new cube with a different fill value.
        """
        return self._fill_value

    @cube_utils.slice_syntax
    def filled_data(self, view):
        """
        Return a portion of the data array, with excluded mask values
        replaced by `fill_value`.

        Returns
        -------
        data : Quantity
            The masked data.
        """
        return u.Quantity(self._get_filled_data(view, fill=self._fill_value),
                          self.unit, copy=False)

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

        """
        from .spectral_axis import convert_spectral_axis,determine_ctype_from_vconv

        if velocity_convention in DOPPLER_CONVENTIONS:
            velocity_convention = DOPPLER_CONVENTIONS[velocity_convention]

        meta = self._meta.copy()
        if 'Original Unit' not in self._meta:
            meta['Original Unit'] = self._wcs.wcs.cunit[self._wcs.wcs.spec]
            meta['Original Type'] = self._wcs.wcs.ctype[self._wcs.wcs.spec]

        out_ctype = determine_ctype_from_vconv(self._wcs.wcs.ctype[self._wcs.wcs.spec],
                                               unit,
                                               velocity_convention=velocity_convention)

        newwcs = convert_spectral_axis(self._wcs, unit, out_ctype,
                                       rest_value=rest_value)

        newmask = self._mask.with_spectral_unit(unit,
                                                velocity_convention=velocity_convention,
                                                rest_value=rest_value)
        newmask._wcs = newwcs

        cube = self._new_cube_with(wcs=newwcs, mask=newmask, meta=meta, spectral_unit=unit)

        return cube

    def _get_filled_data(self, view=(), fill=np.nan, check_endian=False):
        """
        Return the underlying data as a numpy array.
        Always returns the spectral axis as the 0th axis

        Sets masked values to *fill*
        """
        if check_endian:
            if not self._data.dtype.isnative:
                kind = str(self._data.dtype.kind)
                sz = str(self._data.dtype.itemsize)
                dt = '=' + kind + sz
                data = self._data.astype(dt)
            else:
                data = self._data
        else:
            data = self._data

        if self._mask is None:
            return data[view]

        return self._mask._filled(data=data, wcs=self._wcs, fill=fill,
                                  view=view)

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

    @property
    def wcs(self):
        """
        The WCS describing the cube
        """
        return self._wcs

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

        x = x.reshape(1, x.shape[0], x.shape[1])
        y = y.reshape(1, y.shape[0], y.shape[1])
        spectral = spectral.reshape(-1, 1, 1) - spectral.ravel()[0]
        x, y, spectral = np.broadcast_arrays(x, y, spectral)

        return spectral, y, x

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

        dspectral = np.diff(spectral)

        dx = np.abs(np.degrees(dx.reshape(1, dx.shape[0], dx.shape[1])))
        dy = np.abs(np.degrees(dy.reshape(1, dy.shape[0], dy.shape[1])))
        dspectral = np.abs(dspectral.reshape(-1, 1, 1))
        dx, dy, dspectral = np.broadcast_arrays(dx, dy, dspectral)

        # Take spectral units into account
        dspectral = dspectral * self._spectral_scale

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

        .. math:: M_N = \\frac{\\int I (l - M1)**N dl}{M_0}

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

        return Projection(out, copy=False, wcs=new_wcs, meta=meta)

    def moment0(self, axis=0, how='auto'):
        """Compute the zeroth moment along an axis.
        See :meth:`moment`.
        """
        return self.moment(axis=axis, order=0, how=how)

    def moment1(self, axis=0, how='auto'):
        """
        Compute the 1st moment along an axis.
        See :meth:`moment`
        """
        return self.moment(axis=axis, order=1, how=how)

    def moment2(self, axis=0, how='auto'):
        """
        Compute the 2nd moment along an axis.
        See :meth:`moment`
        """
        return self.moment(axis=axis, order=2, how=how)

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

    @property
    def spatial_coordinate_map(self):
        return self.world[0, :, :][1:]

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
                    raise u.UnitsError("Unexpected spectral axis units: {0}".format(spectal_axis.unit))
            elif value.unit.is_equivalent(u.m / u.s):
                if spectral_axis.unit.is_equivalent(u.Hz, equivalencies=u.spectral()):
                    raise u.UnitsError("Spectral axis is in frequency-equivalent "
                                       "units and 'value' is in velocity units "
                                       "- use SpectralCube.with_spectral_unit "
                                       "first to convert the cube to frequency-"
                                       "equivalent units, or search for a "
                                       "velocity instead")
                else:
                    raise u.UnitsError("Unexpected spectral axis units: {0}".format(spectal_axis.unit))
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
                warnings.warn("Mask slicing not implemented for {0} - dropping mask".format(self._mask.__class__.__name__))
                mask_slab = None

        # Create new spectral cube
        slab = self._new_cube_with(data=self._data[ilo:ihi], wcs=wcs_slab, mask=mask_slab)

        # TODO: we could change the WCS to give a spectral axis in the
        # correct units as requested - so if the initial cube is in Hz and we
        # request a range in km/s, we could adjust the WCS to be in km/s
        # instead

        return slab

    def subcube(self, xlo, xhi, ylo, yhi, zlo, zhi, rest_value=None):
        """
        Extract a sub-cube spatially and spectrally.

        This method is not yet implemented.

        xlo = 'min' / 'max' should be special keywords

        """
        raise NotImplementedError()

    def world_spines(self):
        """
        Returns a list of 1D arrays, for the world coordinates
        along each pixel axis.

        Raises error if this operation is ill-posed (e.g. rotated world coordinates,
        strong distortions)

        This method is not currently implemented. Use :meth:`world` instead.
        """
        raise NotImplementedError()

    @cube_utils.slice_syntax
    def world(self, view):
        """
        Return a list of the world coordinates in a cube (or a view of it).

        Cube.world is called with *bracket notation*, like a NumPy array::
            c.world[0:3, :, :]

        Returns
        -------
        [v, y, x] : list of NumPy arryas
            The 3 world coordinates at each pixel in the view.

        Examples
        --------
        >>> c = SpectralCube.read('xyv.fits')

        Extract the first 3 velocity channels of the cube:
        >>> v, y, x = c.world[0:3]

        Extract all the world coordinates
        >>> v, y, x = c.world[:, :, :]

        Extract every other pixel along all axes
        >>> v, y, x = c.world[::2, ::2, ::2]
        """

        # note: view is a tuple of view

        # the next 3 lines are equivalent to (but more efficient than)
        # inds = np.indices(self._data.shape)
        # inds = [i[view] for i in inds]
        inds = np.ogrid[[slice(0, s) for s in self._data.shape]]
        inds = np.broadcast_arrays(*inds)
        inds = [i[view] for i in inds[::-1]]  # numpy -> wcs order

        shp = inds[0].shape
        inds = np.column_stack([i.ravel() for i in inds])
        world = self._wcs.all_pix2world(inds, 0).T

        world = [w.reshape(shp) for w in world]  # 1D->3D

        # apply units
        world = [w * u.Unit(self._wcs.wcs.cunit[i])
                 for i, w in enumerate(world)]

        # convert spectral unit if needed
        if self._spectral_unit is not None:
            world[2] = world[2].to(self._spectral_unit)

        return world[::-1]  # reverse WCS -> numpy order

    def __gt__(self, value):
        """
        Return a LazyMask representing the inequality

        Parameters
        ----------
        value : number
            The threshold
        """
        return LazyMask(lambda data: data > value, data=self._data, wcs=self._wcs)

    def __ge__(self, value):
        return LazyMask(lambda data: data >= value, data=self._data, wcs=self._wcs)

    def __le__(self, value):
        return LazyMask(lambda data: data <= value, data=self._data, wcs=self._wcs)

    def __lt__(self, value):
        return LazyMask(lambda data: data < value, data=self._data, wcs=self._wcs)

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
        cube = read(filename, format=format, hdu=hdu, **kwargs)
        if isinstance(cube, StokesSpectralCube):
            return SpectralCube(data=cube._data, wcs=cube._wcs,
                                meta=cube._meta, mask=cube._mask)
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
            If True, overwrite `filename` if it exists
        """
        from .io.core import write
        write(filename, self, overwrite=overwrite, format=format)

    def to_yt(self, spectral_factor=1.0, nprocs=None, **kwargs):
        """
        Convert a spectral cube to a yt object that can be further analyzed in yt.

        Parameters
        ----------
        spectral_factor : float, optional
            Factor by which to stretch the spectral axis. If set to 1, one pixel
            in spectral coordinates is equivalent to one pixel in spatial
            coordinates.

        If using yt 3.0 or later, additional keyword arguments will be passed onto yt's
        ``FITSDataset`` constructor. See the yt documentation for details on options
        for reading FITS data.
        """

        import yt
        yt_version = float(yt.__version__.split("-")[0])

        if yt_version >= 3.0:

            from yt.frontends.fits.api import FITSDataset
            from astropy.io import fits

            hdu = fits.PrimaryHDU(self._get_filled_data(fill=0.),
                                  header=self.wcs.to_header())

            hdu.header["BUNIT"] = str(self.unit.to_string(format='fits'))
            hdu.header["BTYPE"] = "flux"

            ds = FITSDataset(hdu, nprocs=nprocs, spectral_factor=spectral_factor, **kwargs)

        else:

            from yt.mods import load_uniform_grid

            data = {'flux': self._get_filled_data(fill=0.).transpose()}

            nz, ny, nx = self.shape

            if nprocs is None: nprocs = 1

            bbox = np.array([[0.5,float(nx)+0.5],
                             [0.5,float(ny)+0.5],
                             [0.5,spectral_factor*float(nz)+0.5]])

            ds = load_uniform_grid(data, [nx,ny,nz], 1., bbox=bbox,
                                   nprocs=nprocs, periodicity=(False, False, False))

        return ds

    @property
    def header(self):
        header = self.wcs.to_header()
        header['BUNIT'] = self.unit.to_string(format='fits')
        # TODO: incorporate other relevant metadata here
        return header

    @property
    def hdu(self):
        """
        HDU version of self
        """
        from astropy.io import fits
        hdu = fits.PrimaryHDU(self.filled_data[:].value, header=self.header)
        return hdu

    def world2yt(self, world_coord, spectral_factor=1.0):
        """
        Convert a position in world coordinates to the coordinates used by a
        yt dataset that has been generated using the ``to_yt`` method. The
        changing of the aspect of the spectral axis is handled via the parameter
        ``spectral_factor``.
        """
        yt_coord = self.wcs.wcs_world2pix([world_coord], 1)[0]
        yt_coord[2] = (yt_coord[2] - 0.5)*spectral_factor+0.5
        return yt_coord

    def yt2world(self, yt_coord, spectral_factor=1.0):
        """
        Convert a position in yt's coordinates to world coordinates from a
        yt dataset that has been generated using the ``to_yt`` method. The
        changing of the aspect of the spectral axis is handled via the parameter
        ``spectral_factor``.
        """
        yt_coord = np.array(yt_coord) # stripping off units
        yt_coord[2] = (yt_coord[2] - 0.5)/spectral_factor+0.5
        world_coord = self.wcs.wcs_pix2world([yt_coord], 1)[0]
        return world_coord

class StokesSpectralCube(SpectralCube):

    """
    A class to store a spectral cube with multiple Stokes parameters. By
    default, this will act like a Stokes I spectral cube, but other stokes
    parameters can be accessed with attribute notation.
    """

    def __init__(self, data, wcs, mask=None, meta=None):

        # WCS should be 3-d, data should be dict of 3-d, mask should be disk
        # of 3-d

        # XXX: For now, let's just extract I and work with that

        super(StokesSpectralCube, self).__init__(data["I"], wcs, mask=mask["I"], meta=meta)

        # TODO: deal with the other stokes parameters here

    @classmethod
    def read(cls, filename, format=None, hdu=None):
        """
        Read a spectral cube from a file.

        If the file contains Stokes axes, they will be read in. If you are
        only interested in the unpolarized emission (I), you can use
        :meth:`~spectral_cube.SpectralCube.read` instead.

        Parameters
        ----------
        filename : str
            The file to read the cube from
        format : str
            The format of the file to read. (Currently limited to 'fits' and 'casa_image')
        hdu : int or str
            For FITS files, the HDU to read in (can be the ID or name of an
            HDU).
        """
        raise NotImplementedError("")

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
            If True, overwrite `filename` if it exists
        """
        raise NotImplementedError("")
