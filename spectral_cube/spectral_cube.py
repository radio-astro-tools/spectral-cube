"""
A class to represent a 3-d position-position-velocity spectral cube.
"""

import abc
import warnings

from astropy import units as u
import numpy as np

from . import cube_utils
from . import wcs_utils

__all__ = ['SpectralCube']

try:  # TODO replace with six.py
    xrange
except NameError:
    xrange = range


class MaskBase(object):

    __metaclass__ = abc.ABCMeta

    def include(self, data=None, wcs=None, slices=()):
        """
        Return a boolean array indicating which values should be included.

        If ``slices`` is passed, only the sliced mask will be returned, which
        avoids having to load the whole mask in memory. Otherwise, the whole
        mask is returned in-memory.
        """
        self._validate_wcs(data, wcs)
        return self._include(data=data, wcs=wcs, slices=slices)

    def _validate_wcs(self, data, wcs):
        """
        This method can be overridden in cases where the data and WCS have to
        conform to some rules. This gets called automatically when
        ``include`` or ``exclude`` are called.
        """
        pass

    @abc.abstractmethod
    def _include(self, data=None, wcs=None, slices=()):
        pass

    def exclude(self, data=None, wcs=None, slices=()):
        """
        Return a boolean array indicating which values should be excluded.

        If ``slices`` is passed, only the sliced mask will be returned, which
        avoids having to load the whole mask in memory. Otherwise, the whole
        mask is returned in-memory.
        """
        self._validate_wcs(data, wcs)
        return self._exclude(data=data, wcs=wcs, slices=slices)

    def _exclude(self, data=None, wcs=None, slices=()):
        return ~self._include(data=data, wcs=wcs, slices=slices)

    def _flattened(self, data, wcs=None, slices=()):
        """
        Return a flattened array of the included elements of cube

        Parameters
        ----------
        data : array-like
            The data array to flatten
        slices : tuple, optional
            Any slicing to apply to the data before flattening

        Returns
        -------
        flat_array : `~numpy.ndarray`
            A 1-D ndarray containing the flattened output

        Notes
        -----
        This is an internal method used by :class:`SpectralCube`.
        """
        return data[slices][self.include(data=data, wcs=wcs, slices=slices)]

    def _filled(self, data, wcs=None, fill=np.nan, slices=()):
        """
        Replace the exluded elements of *array* with *fill*.

        Parameters
        ----------
        data : array-like
            Input array
        fill : number
            Replacement value
        slices : tuple, optional
            Any slicing to apply to the data before flattening

        Returns
        -------
        filled_array : `~numpy.ndarray`
            A 1-D ndarray containing the filled output

        Notes
        -----
        This is an internal method used by :class:`SpectralCube`.
        Users should use :meth:`SpectralCubeMask.get_data`
        """
        sliced_data = data[slices]
        return np.where(self.include(data=data, wcs=wcs, slices=slices), sliced_data, fill)

    def __and__(self, other):
        return CompositeMask(self, other, operation='and')

    def __or__(self, other):
        return CompositeMask(self, other, operation='or')

    def __invert__(self):
        return InvertedMask(self)


class InvertedMask(MaskBase):

    def __init__(self, mask):
        self._mask = mask

    def _include(self, data=None, wcs=None, slices=()):
        return ~self._mask.include(data=data, wcs=wcs, slices=slices)


class CompositeMask(MaskBase):
    """
    A combination of several masks. This does an 'and' operation on the
    include masks.
    """

    def __init__(self, mask1, mask2, operation='and'):
        self._mask1 = mask1
        self._mask2 = mask2
        self._operation = operation

    def _validate_wcs(self, new_data, new_wcs):
        self._mask1._validate_wcs(new_data, new_wcs)
        self._mask2._validate_wcs(new_data, new_wcs)

    def _include(self, data=None, wcs=None, slices=()):
        result_mask_1 = self._mask1._include(data=data, wcs=wcs, slices=slices)
        result_mask_2 = self._mask2._include(data=data, wcs=wcs, slices=slices)
        if self._operation == 'and':
            return result_mask_1 & result_mask_2
        elif self._operation == 'or':
            return result_mask_1 | result_mask_2
        else:
            raise ValueError("Operation '{0}' not supported".format(self._operation))

class SpectralCubeMask(MaskBase):
    """
    A mask defined as an array on a spectral cube WCS
    """

    def __init__(self, mask, wcs, include=True):
        self._mask = mask
        self._mask_type = 'include' if include else 'exclude'
        self._wcs = wcs

    def _validate_wcs(self, new_data, new_wcs):
        if new_data.shape != self._mask.shape:
            raise ValueError("data shape does not match mask shape")
        if str(new_wcs.to_header_string()) != str(self._wcs.to_header_string()):
            raise ValueError("WCS does not match mask WCS")

    def _include(self, data=None, wcs=None, slices=()):
        result_mask = self._mask[slices]
        return result_mask if self._mask_type == 'include' else ~result_mask

    def _exclude(self, data=None, wcs=None, slices=()):
        result_mask = self._mask[slices]
        return result_mask if self._mask_type == 'exclude' else ~result_mask

    @property
    def shape(self):
        return self._include_mask.shape

    def __getitem__(self, view):
        return SpectralCubeMask(self._mask[view], wcs_utils.slice_wcs(self._wcs, view))


class LazyMask(MaskBase):
    """
    A boolean mask defined by the evaluation of a function on a fixed dataset.

    This is conceptually identical to a fixed boolean mask as in
    :class:`~spectral_cube.spectralSpectralCubeMask` but defers the
    evaluation of the mask until it is needed.

    Parameters
    ----------
    function : callable
        The function to apply to ``data``
    data : array-like
        The array to evaluate ``function`` on. This should support Numpy-like
        slicing syntax.
    wcs : `~astropy.wcs.WCS`
        The WCS of the input data, which is used to define the coordinates
        for which the boolean mask is defined.
    """

    def __init__(self, function, cube):
        self._function = function
        self._data = cube._data
        self._wcs = cube._wcs

    def _validate_wcs(self, new_data, new_wcs):
        if new_data.shape != self._data.shape:
            raise ValueError("data shape does not match mask shape")
        if str(new_wcs.to_header_string()) != str(self._wcs.to_header_string()):
            raise ValueError("WCS does not match mask WCS")

    def _include(self, data=None, wcs=None, slices=()):
        self._validate_wcs(data, wcs)
        return self._function(self._data[slices])

    def __getitem__(self, view):
        return LazyMask(self._function, self._data[view], wcs_utils.slice_wcs(self._wcs, view))


class FunctionMask(MaskBase):
    """
    A mask defined by a function that is evaluated at run-time using the data
    passed to the mask.

    This is different from :class:`~spectral_cube.spectral_cube.LazyMask` in
    that the mask here can be evaluated using the data passed to the mask and
    the :class:`~spectral_cube.spectral_cube.LazyMask` is applied only to the
    data passed when initializing the
    :class:`~spectral_cube.spectral_cube.LazyMask` instance.

    Parameters
    ----------
    function : callable
        The function to evaluate the mask. The call signature should be
        ``function(data, wcs, slice)`` where ``data`` and ``wcs`` are the
        arguments that get passed to e.g. ``include``, ``exclude``,
        ``_filled``, and ``_flattened``.
    """

    def __init__(self, function):
        self._function = function

    def _validate_wcs(self, data, wcs):
        pass

    def _include(self, data=None, wcs=None, slices=()):
        result = self._function(data, wcs, slices)
        if result.shape != data[slices].shape:
            raise ValueError("Function did not return mask with correct shape - expected {0}, got {1}".format(data[slices].shape, result.shape))
        return result

    def __getitem__(self, slice):
        return self


class SpectralCube(object):

    def __init__(self, data, wcs, mask=None, meta=None):
        # TODO: mask should be oriented? Or should we assume correctly oriented here?
        self._data, self._wcs = cube_utils._orient(data, wcs)
        self._spectral_axis = None
        self._mask = mask  # specifies which elements to Nan/blank/ignore -> SpectralCubeMask
                           # object or array-like object, given that WCS needs to be consistent with data?
        #assert mask._wcs == self._wcs
        self._meta = meta or {}

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    # This should just be relegated to subcube
    # def __getitem__(self, slice):
    # TODO: need to update WCS!
    #    return SpectralCube(self._data[slice], self._wcs,
    #                        mask=self._mask[slice], meta=self.meta)

    def __repr__(self):
        return "SpectralCube with shape {0}: {1}".format(str(self.shape),
                                                         self._data.__repr__())

    def _apply_numpy_function(self, function, fill=np.nan, **kwargs):
        """
        Apply a numpy function to the cube
        """
        return function(self.get_data(fill=fill), **kwargs)

    def get_mask(self):
        return self._mask.include(data=self._data, wcs=self._wcs)

    def sum(self, axis=None):
        # use nansum, and multiply by mask to add zero each time there is badness
        return self._apply_numpy_function(np.nansum, fill=np.nan, axis=axis)

    def max(self, axis=None):
        return self._apply_numpy_function(np.nanmax, fill=np.nan, axis=axis)

    def min(self, axis=None):
        return self._apply_numpy_function(np.nanmin, fill=np.nan, axis=axis)

    def argmax(self, axis=None):
        return self._apply_numpy_function(np.nanargmax, fill=-np.inf, axis=axis)

    def argmin(self, axis=None):
        return self._apply_numpy_function(np.nanargmin, fill=np.inf, axis=axis)

    def chunked(self, chunksize=1000):
        """
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

    def _apply_along_axes(self, function, axis=None, weights=None, wcs=False,
                          **kwargs):
        """
        Apply a function to valid data along the specified axis, optionally
        using a weight array that is the same shape (or at least can be sliced
        in the same way)

        Parameters
        ----------
        function: function
            A function that can be applied to a numpy array.  Does not need to
            be nan-aware
        axis: int
            The axis to operate along
        weights: (optional) np.ndarray
            An array with the same shape (or slicing abilities/results) as the
            data cube
        """
        if axis is None:
            return function(self.flattened(), **kwargs)

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

        if wcs:
            newwcs = wcs_utils.drop_axis(self._wcs, 2-axis)
            return out,newwcs

        return out

    def _iter_rays(self, axis=None):
        """
        Iterate over slices corresponding to lines-of-sight through a cube
        along the specified axis
        """
        nx, ny = self._get_flat_shape(axis)

        for x in xrange(nx):
            for y in xrange(ny):
                # create length-1 slices for each position
                slc = [slice(x, x + 1), slice(y, y + 1)]
                # create a length-N slice (all-inclusive) along the selected axis
                slc.insert(axis, slice(None))
                yield x, y, slc

    def flattened(self, slice=(), weights=None):
        """
        Return a slice of the cube giving only the valid data (i.e., removing
        bad values)

        Parameters
        ----------
        slice: 3-tuple
            A length-3 tuple of slices (or any equivalent valid slice of a
            cube)
        weights: (optional) np.ndarray
            An array with the same shape (or slicing abilities/results) as the
            data cube
        """
        data = self._mask._flattened(data=self._data, wcs=self._wcs, slices=slice)
        if weights is not None:
            weights = self._mask._flattened(data=weights, wcs=self._wcs, slices=slice)
            return data * weights
        else:
            return data

    def median(self, axis=None, **kwargs):
        return self._apply_along_axes(np.median, axis=axis, **kwargs)

    def percentile(self, q, axis=None, **kwargs):
        return self._apply_along_axes(np.percentile, q=q, axis=axis, **kwargs)

    # probably do not want to support this
    # def get_masked_array(self):
    #    return np.ma.masked_where(self.mask, self._data)

    def apply_mask(self, mask, inherit_mask=True):
        """
        Return a new SpectralCube instance that contains a composite mask of
        the current SpectralCube and the new ``mask``.
        """
        cube = SpectralCube(self._data, wcs=self._wcs,
                            mask=self._mask & mask if inherit_mask else mask,
                            meta=self._meta)
        return cube

    def get_filled_data(self, fill=np.nan):
        """
        Return the underlying data as a numpy array.
        Always returns the spectral axis as the 0th axis

        Sets masked values to *fill*
        """
        if self._mask is None:
            return self._data

        return self._mask._filled(data=self._data, wcs=self._wcs, fill=fill)

    def get_unmasked_data(self, copy=False):
        """
        Like data, but don't apply the mask
        """
        if copy:
            return self._data.copy()
        else:
            return self._data

    @property
    def wcs(self):
        return self._wcs

    def moment(self, order, axis, wcs=False):
        """
        Determine the n'th moment along the spectral axis

        If *wcs = True*, return the WCS describing the moment
        """

        nx, ny = self._get_flat_shape(axis)

        # allocate memory for output array
        # nan is a workaround to deal with the impossibility of assigning nan
        # to a np united array
        out = (np.zeros([nx, ny])*np.nan) * u.Unit(self._wcs.wcs.cunit[2 - axis]) ** order

        for x,y,slc in self._iter_rays(axis):
            # the intensity, i.e. the weights
            data = self.flattened(slc)

            # cheat a little if order == 0
            if order == 0:
                weighted = data
                denom = len(data)
            else:
                # compute the world coordinates along the specified axis
                coords = self.world[slc][axis]
                boolmask = self._mask.include(data=self._data, wcs=self._wcs)[slc]
                # the numerator of the moment sum
                weighted = (data*coords[boolmask]**order)
                denom = data.sum()

            # otherwise, leave as nan
            if denom != 0:
                out[x,y] = weighted.sum()/denom

        if wcs:
            newwcs = wcs_utils.drop_axis(self._wcs, 2-axis)
            return out, newwcs
        return out

    def _moment_in_memory(self, order, axis):
        """
        Compute the moments by holding the whole array in memory
        """
        includemask = self._mask.include(data=self._data, wcs=self._wcs)
        if np.any(np.isnan(self._data)):
            data = self.filled(fill=0)
            includemask[np.isnan(data)] = False
            data[np.isnan(data)] = 0
        else:
            data = self._data

        if order == 0:
            return (data*includemask).sum(axis=axis) / includemask.sum(axis=axis)
        else:
            if axis == 0:
                coords = self.spectral_axis[:,None,None]
            else:
                center = self._wcs.wcs.crval[1::-1]
                # this line is wrong; the coordinates have nothing to do with
                # pixel sizes
                mapcoords = (((self.spatial_coordinate_map -
                               center[:,None,None])**2).sum(axis=0)**0.5)
                coords = mapcoords[None,:,:]
            #coords = self.world[:,:,:][axis] * includemask
            mdata = data*includemask
            weighted = (mdata*coords**order)
            denom = mdata.sum(axis=axis)
            return weighted.sum(axis=axis)/denom

    @property
    def spectral_axis(self):
        """
        A `~astropy.units.Quantity` array containing the central values of
        each channel along the spectral axis.
        """
        return self.world[:, 0, 0][0].ravel()

    @property
    def spatial_coordinate_map(self):
        return self.world[0, :, :][1:]

    def closest_spectral_channel(self, value, rest_frequency=None):
        """
        Find the index of the closest spectral channel to the specified
        spectral coordinate.

        Parameters
        ----------
        value : :class:`~astropy.units.Quantity`
            The value of the spectral coordinate to search for.
        rest_frequency : :class:`~astropy.units.Quantity`
            The rest frequency for any Doppler conversions
        """

        # TODO: we have to not compute this every time
        spectral_axis = self.spectral_axis

        try:
            value = value.to(spectral_axis.unit, equivalencies=u.spectral())
        except u.UnitsError:
            if rest_frequency is None:
                raise u.UnitsError("{0} cannot be converted to {1} without a "
                                   "rest frequency".format(value.unit, spectral_axis.unit))
            else:
                try:
                    value = value.to(spectral_axis.unit,
                                     equivalencies=u.doppler_radio(rest_frequency))
                except u.UnitsError:
                    raise u.UnitsError("{0} cannot be converted to {1}".format(value.unit, spectral_axis.unit))

        # TODO: optimize the next line - just brute force for now
        return np.argmin(np.abs(spectral_axis - value))

    def spectral_slab(self, lo, hi, rest_frequency=None):
        """
        Extract a new cube between two spectral coordinates

        Parameters
        ----------
        lo, hi : :class:`~astropy.units.Quantity`
            The lower and upper spectral coordinate for the slab range
        rest_frequency : :class:`~astropy.units.Quantity`
            The rest frequency for any Doppler conversions
        """

        # Find range of values for spectral axis
        ilo = self.closest_spectral_channel(lo, rest_frequency=rest_frequency)
        ihi = self.closest_spectral_channel(hi, rest_frequency=rest_frequency)

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
                mask_slab = self._mask[ilo:ihi,:,:]
            except TypeError:
                warnings.warn("mask slab has not been computed correctly")
                mask_slab = None

        # Create new spectral cube
        slab = SpectralCube(self._data[ilo:ihi], wcs_slab,
                            mask=mask_slab, meta=self._meta)

        # TODO: we could change the WCS to give a spectral axis in the
        # correct units as requested - so if the initial cube is in Hz and we
        # request a range in km/s, we could adjust the WCS to be in km/s
        # instead

        return slab

    def subcube(self, xlo, xhi, ylo, yhi, zlo, zhi, rest_frequency=None):
        """
        Extract a sub-cube spatially and spectrally

        xlo = 'min' / 'max' should be special keywords
        """

    def world_spines(self):
        """
        Returns a dict of 1D arrays, for the world coordinates
        along each pixel axis.

        Raises error if this operation is ill-posed (e.g. rotated world coordinates,
        strong distortions)
        """

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
        >>> c = read('xyv.fits')

        Extract the first 3 velocity channels of the cube:
        >>> v, y, x = c.world[0:3]

        Extract all the world coordinates
        >>> v, y, x = c.world[:, :, :]

        Extract every other pixel along all axes
        >>> v, y, x = c.world[::2, ::2, ::2]

        Note
        ----
        Calling world with slices is efficient in the sense that it
        only computes pixels within the view.
        """

        # note: view is a tuple of slices

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
        return world[::-1]  # reverse WCS -> numpy order


    def write(self, filename, format=None, include_stokes=False, clobber=False):
        if format == 'fits':
            from .io.fits import write_fits_cube
            write_fits_cube(filename, self,
                            include_stokes=include_stokes, clobber=clobber)
        else:
            raise NotImplementedError("Try FITS instead")

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


def read(filename, format=None):
    if format == 'fits':
        from .io.fits import load_fits_cube
        return load_fits_cube(filename)
    elif format == 'casa_image':
        from .io.casa_image import load_casa_image
        return load_casa_image(filename)
    else:
        raise ValueError("Format {0} not implemented".format(format))
