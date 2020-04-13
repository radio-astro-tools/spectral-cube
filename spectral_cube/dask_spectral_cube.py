"""
A class to represent a 3-d position-position-velocity spectral cube.
"""

from __future__ import print_function, absolute_import, division

import warnings
from functools import wraps

from astropy import units as u
from astropy.io.fits import PrimaryHDU, HDUList
from astropy.wcs.utils import proj_plane_pixel_area

import numpy as np
import dask.array as da

from astropy import stats
from astropy import convolution
from astropy import wcs

from . import wcs_utils
from .spectral_cube import SpectralCube, VaryingResolutionSpectralCube, SIGMA2FWHM, np2wcs
from .utils import cached, warn_slow, VarianceWarning, SliceWarning, BeamWarning
from .lower_dimensional_structures import Projection
from .masks import BooleanArrayMask, is_broadcastable_and_smaller
from .np_compat import allbadtonan

__all__ = ['DaskSpectralCube', 'DaskVaryingResolutionSpectralCube']

try:
    from scipy import ndimage
    SCIPY_INSTALLED = True
except ImportError:
    SCIPY_INSTALLED = False


def nansum_allbadtonan(dask_array, axis=None, keepdims=None):
    return da.reduction(dask_array,
                        allbadtonan(np.nansum),
                        allbadtonan(np.nansum),
                        axis=axis,
                        dtype=dask_array.dtype)


def projection_if_needed(function):

    @wraps(function)
    def wrapper(self, *args, **kwargs):

        out = function(self, *args, **kwargs)

        axis = kwargs.get('axis')

        if axis is not None and self._naxes_dropped(axis) in (1, 2):

            meta = {'collapse_axis': axis}
            meta.update(self._meta)

            if hasattr(axis, '__len__') and len(axis) == 2:
                # if operation is over two spatial dims
                if set(axis) == set((1, 2)):
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
                                               unit=self.unit,
                                               header=header,
                                               meta=meta,
                                               spectral_unit=self._spectral_unit,
                                               **bmarg
                                              )
                else:
                    warnings.warn("Averaging over a spatial and a spectral "
                                  "dimension cannot produce a Projection "
                                  "quantity (no units or WCS are preserved).",
                                  SliceWarning)
                    return out

            else:
                new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])
                header = self._nowcs_header

            return Projection(out, copy=False, wcs=new_wcs,
                              meta=meta,  unit=self.unit,
                              header=header)

        else:

            return u.Quantity(out, unit=self.unit)

    return wrapper


def ignore_deprecated_kwargs(function):

    @wraps(function)
    def wrapper(self, *args, **kwargs):

        if 'how' in kwargs:
            kwargs.pop('how')
            warnings.warn('The how= argument is ignored by methods of {0} '
                          'since how the operations are carried out is '
                          'controlled by the chunking of the underlying dask '
                          'array.'.format(self.__class__.__name__))

        if 'iterate_rays' in kwargs:
            kwargs.pop('iterate_rays')
            warnings.warn('The iterate_rays= argument is ignored by methods of {0} '
                          'since how the operations are carried out is '
                          'controlled by the chunking of the underlying dask '
                          'array.'.format(self.__class__.__name__))

        if 'progressbar' in kwargs:
            kwargs.pop('progressbar')
            warnings.warn('The progressbar= argument is ignored by methods of {0} '
                          'since you can instead use dask\'s mechanism to '
                          'create one.'.format(self.__class__.__name__))

        if 'update_function' in kwargs:
            kwargs.pop('update_function')
            warnings.warn('The update_function= argument is ignored by methods of {0} '
                          'since you can instead use dask\'s mechanism to '
                          'create one.'.format(self.__class__.__name__))

        return function(self, *args, **kwargs)

    return wrapper


class DaskSpectralCubeMixin:

    def _compute(self, array):
        # For now, always default to serial mode, but we could then expand this
        # to allow different modes.
        return array.compute(scheduler='synchronous')

    def _filled_dask_array(self, fill=None):
        if self._mask is None:
            return self._data
        else:
            return da.asarray(self._mask._filled(data=self._data,
                                                 wcs=self._wcs, fill=fill or self.fill_value,
                                                 wcs_tolerance=self._wcs_tolerance))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
    def sum(self, axis=None, **kwargs):
        """
        Return the sum of the cube, optionally over an axis.
        """
        return self._compute(nansum_allbadtonan(self._filled_dask_array(fill=np.nan), axis=axis, **kwargs))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
    def mean(self, axis=None, **kwargs):
        """
        Return the mean of the cube, optionally over an axis.
        """
        return self._compute(da.nanmean(self._filled_dask_array(fill=np.nan), axis=axis, **kwargs))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
    def median(self, axis=None, **kwargs):
        """
        Return the median of the cube, optionally over an axis.
        """
        data = self._filled_dask_array(fill=np.nan)

        if axis is None:
            return np.nanmedian(data.compute(), **kwargs)
        else:
            return self._compute(da.nanmedian(self._filled_dask_array(fill=np.nan), axis=axis, **kwargs))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
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

        data = self._filled_dask_array(fill=np.nan)

        if axis is None:
            return np.nanpercentile(data, q, **kwargs)
        else:
            # Rechunk so that there is only one chunk along the desired axis
            data = data.rechunk([-1 if i == axis else 'auto' for i in range(3)])
            return self._compute(data.map_blocks(np.nanpercentile, q=q, drop_axis=axis, axis=axis, **kwargs))

        return self._compute(da.nanpercentile(self._filled_dask_array(fill=np.nan), q, axis=axis))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
    def std(self, axis=None, ddof=0, **kwargs):
        """
        Return the mean of the cube, optionally over an axis.

        Other Parameters
        ----------------
        ddof : int
            Means Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.  By
            default ``ddof`` is zero.
        """
        return self._compute(da.nanstd(self._filled_dask_array(fill=np.nan), axis=axis, ddof=ddof, **kwargs))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
    def mad_std(self, axis=None, **kwargs):
        """
        Use astropy's mad_std to compute the standard deviation
        """

        data = self._filled_dask_array(fill=np.nan)

        if axis is None:
            # In this case we have to load the full data - even dask's
            # nanmedian doesn't work efficiently over the whole array.
            return stats.mad_std(data, **kwargs)
        else:
            # Rechunk so that there is only one chunk along the desired axis
            data = data.rechunk([-1 if i == axis else 'auto' for i in range(3)])
            return self._compute(data.map_blocks(stats.mad_std, drop_axis=axis, axis=axis, **kwargs))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
    def max(self, axis=None, **kwargs):
        """
        Return the maximum data value of the cube, optionally over an axis.
        """
        return self._compute(da.nanmax(self._filled_dask_array(fill=np.nan), axis=axis, **kwargs))

    @warn_slow
    @projection_if_needed
    @ignore_deprecated_kwargs
    def min(self, axis=None, **kwargs):
        """
        Return the minimum data value of the cube, optionally over an axis.
        """
        return self._compute(da.nanmin(self._filled_dask_array(fill=np.nan), axis=axis, **kwargs))

    @warn_slow
    @ignore_deprecated_kwargs
    def argmax(self, axis=None, **kwargs):
        """
        Return the index of the maximum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return self._compute(da.nanargmax(self._filled_dask_array(fill=-np.inf), axis=axis, **kwargs))

    @warn_slow
    @ignore_deprecated_kwargs
    def argmin(self, axis=None, **kwargs):
        """
        Return the index of the minimum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return self._compute(da.nanargmin(self._filled_dask_array(fill=np.inf), axis=axis))

    def _map_blocks_to_cube(self, function, additional_arrays=None, rechunk=None, **kwargs):
        """
        Call dask's map_blocks, returning a new spectral cube.
        """

        if rechunk is None:
            data = self._filled_dask_array(fill=np.nan)
        else:
            data = self._filled_dask_array(fill=np.nan).rechunk(rechunk)

        if additional_arrays is None:
            newdata = data.map_blocks(function, dtype=data.dtype, **kwargs)
        else:
            newdata = da.map_blocks(function, data, *additional_arrays, dtype=data.dtype, **kwargs)

        # Create final output cube
        newcube = self._new_cube_with(data=newdata,
                                      wcs=self.wcs,
                                      mask=self.mask,
                                      meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube

    def sigma_clip_spectrally(self,
                              threshold,
                              verbose=0,
                              num_cores=None,
                              **kwargs):
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

        def spectral_sigma_clip(array):
            result = stats.sigma_clip(array, sigma=threshold, axis=0, **kwargs)
            return result.filled(np.nan)

        # Rechunk so that there is only one chunk spectrally and let dask
        # decide for the rest
        return self._map_blocks_to_cube(spectral_sigma_clip,
                                        rechunk=(-1, 'auto', 'auto'))

    def spectral_smooth(self,
                        kernel,
                        convolve=convolution.convolve,
                        verbose=0,
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

        def spectral_smooth(array):
            if array.size > 0:
                kernel_3d = kernel.array.reshape((len(kernel.array), 1, 1))
                return convolve(array, kernel_3d, normalize_kernel=True)
            else:
                return array

        # Rechunk so that there is only one chunk spectrally
        return self._map_blocks_to_cube(spectral_smooth,
                                        rechunk=(-1, 'auto', 'auto'))

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

        if not SCIPY_INSTALLED:
            raise ImportError("Scipy could not be imported: this function won't work.")

        if float(ksize).is_integer():
            ksize = int(ksize)
        else:
            raise TypeError('ksize should be an integer (got {0})'.format(ksize))

        def median_filter_wrapper(img, **kwargs):
            return ndimage.median_filter(img, (ksize, 1, 1), **kwargs)

        # Rechunk so that there is only one chunk spectrally
        return self._map_blocks_to_cube(median_filter_wrapper,
                                        rechunk=(-1, 'auto', 'auto'))

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

        kernel = kernel.array.reshape((1,) + kernel.array.shape)

        def convolve_wrapper(img, **kwargs):
            if img.size > 0:
                return convolve(img, kernel, normalize_kernel=True, **kwargs)
            else:
                return img

        # Rechunk so that there is only one chunk in the image plane
        return self._map_blocks_to_cube(convolve_wrapper,
                                        rechunk=('auto', -1, -1))

    def spatial_smooth_median(self, ksize, **kwargs):
        """
        Smooth the image in each spatial-spatial plane of the cube using a median filter.

        Parameters
        ----------
        ksize : int
            Size of the median filter (scipy.ndimage.filters.median_filter)
        kwargs : dict
            Passed to the median_filter function
        """

        if not SCIPY_INSTALLED:
            raise ImportError("Scipy could not be imported: this function won't work.")

        def median_filter_wrapper(img, **kwargs):
            return ndimage.median_filter(img, (1, ksize, ksize), **kwargs)

        # Rechunk so that there is only one chunk in the image plane
        return self._map_blocks_to_cube(median_filter_wrapper,
                                        rechunk=('auto', -1, -1))

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

        x, y, spectral = da.broadcast_arrays(x[None,:,:], y[None,:,:], spectral[:,None,None])

        return spectral, y, x

    def moment(self, order=0, axis=0, **kwargs):
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

        Returns
        -------
        map [, wcs]
           The moment map (numpy array) and, if wcs=True, the WCS object
           describing the map

        Notes
        -----
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

        data = self._filled_dask_array(fill=np.nan).astype(np.float64)

        pix_size = self._pix_size_slice(axis)
        pix_cen = self._pix_cen()[axis]

        if order == 0:
            out = nansum_allbadtonan(data * pix_size, axis=axis)
        else:
            mom1 = (nansum_allbadtonan(data * pix_size * pix_cen, axis=axis) /
                    nansum_allbadtonan(data * pix_size, axis=axis))
            if order > 1:
                # insert an axis so it broadcasts properly
                shp = list(mom1.shape)
                shp.insert(axis, 1)
                mom1 = mom1.reshape(shp)
                out = (nansum_allbadtonan(data * pix_size * (pix_cen - mom1) ** order, axis=axis) /
                       nansum_allbadtonan(data * pix_size, axis=axis))
            else:
                out = mom1

        # force computation, and convert back to original dtype (but native)
        out = self._compute(out)

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
                'moment_axis': axis}
        meta.update(self._meta)

        print(out.shape)

        return Projection(out, copy=False, wcs=new_wcs, meta=meta,
                          header=self._nowcs_header)

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

        if isinstance(region_mask, np.ndarray):
            if is_broadcastable_and_smaller(region_mask.shape, self.shape):
                region_mask = BooleanArrayMask(region_mask, self._wcs)
            else:
                raise ValueError("Mask shape does not match cube shape.")

        include = region_mask.include(self._data, self._wcs,
                                      wcs_tolerance=self._wcs_tolerance)

        include = da.broadcast_to(include, self.shape)

        # NOTE: the approach in the base SpectralCube class is incorrect, if
        # there are multiple 'islands' of valid values in the cube, as this will
        # pick only the first one found by find_objects. Here we use a more
        # robust approach.

        slices = []
        for axis in range(3):
            collapse_axes = tuple(index for index in range(3) if index != axis)
            valid = self._compute(da.any(include, axis=collapse_axes))
            if np.any(valid):
                indices = np.where(valid)[0]
                slices.append(slice(np.min(indices), np.max(indices) + 1))
            else:
                slices.append(slice(0))

        if spatial_only:
            slices = (slice(None), slices[1], slices[2])

        return tuple(slices)

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
        # Quantity does not work well with lazily evaluated data with an
        # unkonwn shape (which is the case when doing boolean indexing of arrays)
        if isinstance(data, da.Array):
            data = data.compute()
        if weights is not None:
            weights = self._mask._flattened(data=weights, wcs=self._wcs, view=slice)
            return u.Quantity(data * weights, self.unit, copy=False)
        else:
            return u.Quantity(data, self.unit, copy=False)

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
        """

        data = self._filled_dask_array()
        mask = da.asarray(self.mask.include())

        if not truncate and data.shape[axis] % factor != 0:
            padding_shape = list(data.shape)
            padding_shape[axis] = factor - data.shape[axis] % factor
            data_padding = da.ones(padding_shape) * np.nan
            mask_padding = da.zeros(padding_shape, dtype=bool)
            data = da.concatenate([data, data_padding], axis=axis)
            mask = da.concatenate([mask, mask_padding], axis=axis).rechunk()

        if data.chunksize[axis] % factor != 0:
            chunksize = list(data.chunksize)
            chunksize[axis] = chunksize[axis] + factor - chunksize[axis] % factor
            data = data.rechunk(chunksize)

        if mask.chunksize[axis] % factor != 0:
            chunksize = list(mask.chunksize)
            chunksize[axis] = chunksize[axis] + factor - chunksize[axis] % factor
            mask = mask.rechunk(chunksize)

        data = da.coarsen(estimator, data, {axis: factor}, trim_excess=truncate)
        mask = da.coarsen(estimator, mask, {axis: factor}, trim_excess=truncate)

        view = [slice(None, None, factor) if ii == axis else slice(None) for ii in range(self.ndim)]
        newwcs = wcs_utils.slice_wcs(self.wcs, view, shape=self.shape)
        newwcs._naxis = list(self.shape)

        # this is an assertion to ensure that the WCS produced is valid
        # (this is basically a regression test for #442)
        assert newwcs[:, slice(None), slice(None)]
        assert len(newwcs._naxis) == 3

        return self._new_cube_with(data=data, wcs=newwcs,
                                   mask=BooleanArrayMask(mask, wcs=newwcs))


class DaskSpectralCube(DaskSpectralCubeMixin, SpectralCube):

    @classmethod
    def read(cls, *args, **kwargs):

        cube = super().read(*args, **kwargs)

        cube._data = da.asarray(cube._data)

        if isinstance(cube, VaryingResolutionSpectralCube):
            return DaskVaryingResolutionSpectralCube(cube._data, cube.wcs, mask=cube.mask,
                                                     meta=cube.meta, fill_value=cube.fill_value,
                                                     header=cube.header,
                                                     allow_huge_operations=cube.allow_huge_operations,
                                                     beams=cube.beams, wcs_tolerance=cube._wcs_tolerance)
        else:
            return DaskSpectralCube(cube._data, cube.wcs, mask=cube.mask,
                                    meta=cube.meta, fill_value=cube.fill_value,
                                    header=cube.header,
                                    allow_huge_operations=cube.allow_huge_operations,
                                    beam=cube._beam, wcs_tolerance=cube._wcs_tolerance)

    @property
    def hdu(self):
        """
        HDU version of self
        """
        return PrimaryHDU(self._filled_dask_array(), header=self.header)

    @property
    def hdulist(self):
        return HDUList(self.hdu)

    def convolve_to(self, beam, convolve=convolution.convolve, update_function=None, **kwargs):
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

        pixscale = proj_plane_pixel_area(self.wcs.celestial)**0.5 * u.deg

        convolution_kernel = beam.deconvolve(self.beam).as_kernel(pixscale)
        kernel = convolution_kernel.array.reshape((1,) + convolution_kernel.array.shape)

        def convfunc(img):
            if img.size > 0:
                return convolve(img, kernel, normalize_kernel=True, **kwargs).reshape(img.shape)
            else:
                return img

        # Rechunk so that there is only one chunk in the image plane
        return self._map_blocks_to_cube(convfunc,
                                        rechunk=('auto', -1, -1)).with_beam(beam)


class DaskVaryingResolutionSpectralCube(DaskSpectralCubeMixin, VaryingResolutionSpectralCube):

    @property
    def hdu(self):
        raise ValueError("For DaskVaryingResolutionSpectralCube's, use hdulist "
                         "instead of hdu.")

    @property
    def hdulist(self):
        """
        HDUList version of self
        """

        hdu = PrimaryHDU(self._filled_dask_array(), header=self.header)

        from .cube_utils import beams_to_bintable
        # use unmasked beams because, even if the beam is masked out, we should
        # write it
        bmhdu = beams_to_bintable(self.unmasked_beams)

        return HDUList([hdu, bmhdu])

    def convolve_to(self, beam, allow_smaller=False,
                    convolve=convolution.convolve_fft,
                    update_function=None, **kwargs):
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
        for bm, valid in zip(self.unmasked_beams, self.goodbeams_mask):
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

        # We need to pass in the kernels to dask as an original array - however
        # since each kernel array may be a different size, we hide it inside an
        # object array.
        convolution_kernels = da.from_array(np.array(convolution_kernels, dtype=np.object)
                                            .reshape((len(convolution_kernels), 1, 1)), chunks=(1,-1,-1))

        def convfunc(img, kernel):
            if img.size > 0:
                kernel = kernel[0, 0, 0]
                return convolve(img[0], kernel, normalize_kernel=True, **kwargs).reshape(img.shape)
            else:
                return img

        # Rechunk so that there is only one chunk in the image plane
        cube = self._map_blocks_to_cube(convfunc,
                                        additional_arrays=(convolution_kernels,),
                                        rechunk=(1, -1, -1))

        # Result above is a DaskVaryingResolutionSpectralCube, convert to DaskSpectralCube
        newcube = DaskSpectralCube(data=cube._data,
                                   beam=beam,
                                   wcs=cube.wcs,
                                   mask=cube.mask,
                                   meta=cube.meta,
                                   fill_value=cube.fill_value)

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
