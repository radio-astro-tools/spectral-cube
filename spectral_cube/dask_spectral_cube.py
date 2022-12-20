"""
A class to represent a 3-d position-position-velocity spectral cube.
"""

from __future__ import print_function, absolute_import, division

import uuid
import inspect
import warnings
import tempfile

from functools import wraps
from contextlib import contextmanager

from astropy import units as u
from astropy.io.fits import PrimaryHDU, HDUList
from astropy.wcs.utils import proj_plane_pixel_area

import numpy as np

import dask
import dask.array as da

from astropy import stats
from astropy import convolution
from astropy import wcs

from . import wcs_utils
from .spectral_cube import SpectralCube, VaryingResolutionSpectralCube, SIGMA2FWHM, np2wcs
from .utils import cached, VarianceWarning, SliceWarning, BeamWarning, SmoothingWarning, BeamUnitsError
from .lower_dimensional_structures import Projection
from .masks import BooleanArrayMask, is_broadcastable_and_smaller
from .np_compat import allbadtonan

__all__ = ['DaskSpectralCube', 'DaskVaryingResolutionSpectralCube']

try:
    from scipy import ndimage
    import scipy.interpolate
    SCIPY_INSTALLED = True
except ImportError:
    SCIPY_INSTALLED = False

try:
    import zarr
    import fsspec
except ImportError:
    ZARR_INSTALLED = False
else:
    ZARR_INSTALLED = True


def nansum_allbadtonan(dask_array, axis=None, keepdims=None):
    return da.reduction(dask_array,
                        allbadtonan(np.nansum),
                        allbadtonan(np.nansum),
                        axis=axis,
                        dtype=dask_array.dtype)


def ignore_warnings(function):

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return function(self, *args, **kwargs)

    return wrapper


def add_save_to_tmp_dir_option(function):

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        save_to_tmp_dir = kwargs.pop('save_to_tmp_dir', False)
        cube = function(self, *args, **kwargs)
        if save_to_tmp_dir and isinstance(cube, DaskSpectralCubeMixin):
            if not ZARR_INSTALLED:
                raise ImportError("saving the cube to a temporary directory "
                                  "requires the zarr and fsspec packages to "
                                  "be installed.")
            filename = tempfile.mktemp()
            with dask.config.set(**cube._scheduler_kwargs):
                cube._data.to_zarr(filename)
            cube._data = da.from_zarr(filename)
        return cube

    return wrapper


def projection_if_needed(function):

    # check if function defines default projection kwargs
    parameters = inspect.signature(function).parameters

    if 'projection' in parameters:
        default_projection = parameters['projection'].default
    else:
        default_projection = True

    if 'unit' in parameters:
        default_unit = parameters['unit'].default
    else:
        default_unit = 'self'

    @wraps(function)
    def wrapper(self, *args, **kwargs):

        projection = kwargs.get('projection', default_projection)
        unit = kwargs.get('unit', default_unit)

        if unit == 'self':
            unit = self.unit

        out = function(self, *args, **kwargs)

        axis = kwargs.get('axis')

        if isinstance(out, da.Array):
            out = self._compute(out)

        if axis is None:

            # return is scalar
            if unit is not None:
                return u.Quantity(out, unit=unit)
            else:
                return out

        elif projection and axis is not None and self._naxes_dropped(axis) in (1, 2):

            meta = {'collapse_axis': axis}
            meta.update(self._meta)

            if hasattr(axis, '__len__') and len(axis) == 2:
                # if operation is over two spatial dims
                if set(axis) == set((1, 2)):
                    new_wcs = self._wcs.sub([wcs.WCSSUB_SPECTRAL])
                    header = self._nowcs_header
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
                                  SliceWarning)
                    return out

            else:
                new_wcs = wcs_utils.drop_axis(self._wcs, np2wcs[axis])
                header = self._nowcs_header

                return Projection(out, copy=False, wcs=new_wcs,
                                  meta=meta,  unit=unit,
                                  header=header)

        else:

            return out

    return wrapper


class FilledArrayHandler:
    """
    This class is a wrapper for the data and mask which can be used to
    initialize a dask array. It provides a way for the filled data to be
    constructed just for the requested chunks.
    """

    def __init__(self, cube, fill=np.nan):
        self._data = cube._data
        self._mask = cube._mask
        self._fill = fill
        self._wcs = cube._wcs
        self._wcs_tolerance = cube._wcs_tolerance
        self.shape = cube._data.shape
        self.dtype = cube._data.dtype
        self.ndim = len(self.shape)

    def __getitem__(self, view):
        if self._data[view].size == 0:
            return 0.
        else:
            return self._mask._filled(data=self._data,
                                      view=view,
                                      wcs=self._wcs,
                                      fill=self._fill,
                                      wcs_tolerance=self._wcs_tolerance)


class MaskHandler:
    """
    This class is a wrapper for the mask which can be used to initialize a dask
    array. It provides a way for the mask to be computed just for the requested
    chunk.
    """

    def __init__(self, cube):
        self._data = cube._data
        self._mask = cube.mask
        self.shape = cube._data.shape
        self.dtype = cube._data.dtype
        self.ndim = len(self.shape)

    def __getitem__(self, view):
        if self._data[view].size == 0:
            return False
        else:
            result = self._mask.include(view=view)
            if isinstance(result, da.Array):
                result = result.compute()
        return result


class DaskSpectralCubeMixin:

    _scheduler_kwargs = {'scheduler': 'synchronous'}

    def _new_cube_with(self, *args, **kwargs):
        # The scheduler should be preserved for cubes produced as a result
        # of this one.
        new_cube = super()._new_cube_with(*args, **kwargs)
        new_cube._scheduler_kwargs = self._scheduler_kwargs
        return new_cube

    @property
    def _data(self):
        return self.__data

    @_data.setter
    def _data(self, value):
        if not isinstance(value, da.Array):
            raise TypeError('_data should be set to a dask array')
        self.__data = value

    def use_dask_scheduler(self, scheduler, num_workers=None):
        """
        Set the dask scheduler to use.

        Can be used as a function or a context manager.

        Parameters
        ----------
        scheduler : str
            Any valid dask scheduler. See https://docs.dask.org/en/latest/scheduler-overview.html
            for an overview of available schedulers.
        num_workers : int
            Number of workers to use for the 'threads' and 'processes' schedulers.
        """

        original_scheduler_kwargs = self._scheduler_kwargs
        self._scheduler_kwargs = {'scheduler': scheduler}
        if num_workers is not None:
            self._scheduler_kwargs['num_workers'] = num_workers

        self._num_workers = num_workers

        class SchedulerHandler:

            def __init__(self, cube, original_scheduler_kwargs):
                self.cube = cube
                self.original_scheduler_kwargs = original_scheduler_kwargs

            def __enter__(self):
                pass

            def __exit__(self, *args):
                self.cube._scheduler_kwargs = self.original_scheduler_kwargs

        return SchedulerHandler(self, original_scheduler_kwargs)

    def _compute(self, array):
        return array.compute(**self._scheduler_kwargs)

    def _warn_slow(self, funcname):
        if self._is_huge and not self.allow_huge_operations:
            raise ValueError("This function ({0}) requires loading the entire "
                             "cube into memory, and the cube is large ({1} "
                             "pixels), so by default we disable this operation. "
                             "To enable the operation, set "
                             "`cube.allow_huge_operations=True` and try again."
                             .format(funcname, self.size))

    def _get_filled_data(self, view=(), fill=np.nan, check_endian=None, use_memmap=None):

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
        else:
            return da.from_array(FilledArrayHandler(self, fill=fill), name='FilledArrayHandler ' + str(uuid.uuid4()), chunks=data.chunksize)[view]

    def __repr__(self):
        default_repr = super().__repr__()
        lines = default_repr.splitlines()
        lines[0] = lines[0][:-1] + ' and chunk size {0}:'.format(self._data.chunksize)
        return '\n'.join(lines)

    def display_dask_array(self):
        try:
            from IPython.display import display
            return display(self._data)
        except ImportError:
            warnings.warn("Requires IPython to display.")

    @add_save_to_tmp_dir_option
    def rechunk(self, chunks='auto', threshold=None, block_size_limit=None,
                **kwargs):
        """
        Rechunk the underlying dask array and return a new cube.

        For more details about the parameters below, see the dask documentation
        about `rechunking <https://docs.dask.org/en/latest/array-chunks.html>`_.

        Parameters
        ----------
        chunks:  int, tuple, dict or str, optional
            The new block dimensions to create. -1 indicates the full size of
            the corresponding dimension. Default is "auto" which automatically
            determines chunk sizes. This can also be a tuple with a different
            value along each dimension - for example if computing moment maps,
            you could use e.g. ``chunks=(-1, 'auto', 'auto')``
        threshold: int, optional
            The graph growth factor under which we don't bother introducing an
            intermediate step.
        block_size_limit: int, optional
            The maximum block size (in bytes) we want to produce
            Defaults to the dask configuration value ``array.chunk-size``
        save_to_tmp_dir : bool
            If `True`, the rechunking will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the rechunking is added as a step in the dask tree.
        kwargs
            Additional keyword arguments are passed to the dask rechunk method.
        """

        newdata = self._data.rechunk(chunks=chunks,
                                     threshold=threshold,
                                     block_size_limit=block_size_limit)

        return self._new_cube_with(data=newdata)

    @add_save_to_tmp_dir_option
    @projection_if_needed
    def apply_function(self, function, axis=None, unit=None,
                       projection=False,
                       keep_shape=False, **kwargs):
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
        unit : (optional) `~astropy.units.Unit`
            The unit of the output projection or value.  Not all functions
            should return quantities with units.
        projection : bool
            Return a projection if the resulting array is 2D?
        keep_shape : bool
            If `True`, the returned object will be the same dimensionality as
            the cube.
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.

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

        data = self._get_filled_data(fill=self._fill_value)

        if keep_shape:
            newdata = da.apply_along_axis(function, axis, data, shape=(self.shape[axis],))
        else:
            newdata = da.apply_along_axis(function, axis, data)

        return newdata

    @add_save_to_tmp_dir_option
    @projection_if_needed
    def apply_numpy_function(self, function, fill=np.nan,
                             projection=False,
                             unit=None,
                             check_endian=False,
                             **kwargs):
        """
        Apply a numpy function to the cube

        Parameters
        ----------
        function : Numpy ufunc
            A numpy ufunc to apply to the cube
        fill : float
            The fill value to use on the data
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
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
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

        data = self._get_filled_data(fill=fill, check_endian=check_endian)

        # Numpy ufuncs know how to deal with dask arrays
        if function.__module__.startswith('numpy'):
            return function(data, **kwargs)
        else:
            # TODO: implement support for bottleneck? or arbitrary ufuncs?
            raise NotImplementedError()

    @add_save_to_tmp_dir_option
    def apply_function_parallel_spatial(self,
                                        function,
                                        accepts_chunks=False,
                                        **kwargs):
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
            image.
        accepts_chunks : bool
            Whether the function can take chunks with shape (ns, ny, nx) where
            ``ns`` is the number of spectral channels in the cube and ``nx``
            and ``ny`` may be greater than one.
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
        kwargs : dict
            Passed to ``function``
        """

        if accepts_chunks:
            def wrapper(data_slices, **kwargs):
                if data_slices.size > 0:
                    return function(data_slices, **kwargs)
                else:
                    return data_slices
        else:
            def wrapper(data_slices, **kwargs):
                if data_slices.size > 0:
                    out = np.zeros_like(data_slices)
                    for index in range(data_slices.shape[0]):
                        out[index] = function(data_slices[index], **kwargs)
                    return out
                else:
                    return data_slices

        # Rechunk so that there is only one chunk in the image plane
        return self._map_blocks_to_cube(wrapper,
                                        rechunk=('auto', -1, -1),
                                        fill=self._fill_value, **kwargs)

    @add_save_to_tmp_dir_option
    def apply_function_parallel_spectral(self,
                                         function,
                                         accepts_chunks=False,
                                         return_new_cube=True,
                                         **kwargs):
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
        accepts_chunks : bool
            Whether the function can take chunks with shape (ns, ny, nx) where
            ``ns`` is the number of spectral channels in the cube and ``nx``
            and ``ny`` may be greater than one.
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
        return_new_cube : bool
            If `True`, a new `~SpectralCube` object will be returned. This is the default
            for when the function will return another version of the new spectral cube
            with the operation applied (for example, spectral smoothing). If `False`,
            an array will be returned from `function`. This is useful, for example,
            when fitting a model to spectra and the output is the fitted model parameters.
        kwargs : dict
            Passed to ``function``
        """

        # NOTE: `block_info` should always be available for `dask.array.map_blocks` to pass to
        # Because we use this wrapper, this always should be an available kwarg, then we check
        # if that kwarg should be passed to `function`
        _has_blockinfo = 'block_info' in inspect.signature(function).parameters

        # if/else to avoid an if/else in every single wrapper call.
        if _has_blockinfo:
            def wrapper(data, block_info=None, **kwargs):
                if data.size > 0:
                    return function(data, block_info=block_info, **kwargs)
                else:
                    return data
        else:
            def wrapper(data, **kwargs):
                if data.size > 0:
                    return function(data, **kwargs)
                else:
                    return data

        if accepts_chunks:
            # Check if the spectral axis is already one chunk. If it is, there is no need to rechunk the data
            current_chunksize = self._data.chunksize
            if current_chunksize[0] == self.shape[0]:
                rechunk = None
            else:
                rechunk = (-1, 'auto', 'auto')

            return self._map_blocks_to_cube(wrapper,
                                            return_new_cube=return_new_cube,
                                            rechunk=rechunk, **kwargs)

        else:
            data = self._get_filled_data(fill=self._fill_value)
            # apply_along_axis returns an array with a single chunk, but we
            # need to rechunk here to avoid issues when writing out the data
            # even if it results in a poorer performance.
            data = data.rechunk((-1, 'auto', 'auto'))
            newdata = da.apply_along_axis(wrapper, 0, data, shape=(self.shape[0],),
                                          **kwargs)

            if return_new_cube:
                return self._new_cube_with(data=newdata, wcs=self.wcs,
                                           mask=self.mask, meta=self.meta,
                                           fill_value=self.fill_value)
            else:
                return newdata


    @projection_if_needed
    @ignore_warnings
    def sum(self, axis=None, **kwargs):
        """
        Return the sum of the cube, optionally over an axis.
        """
        return self._compute(nansum_allbadtonan(self._get_filled_data(fill=np.nan), axis=axis, **kwargs))

    @projection_if_needed
    @ignore_warnings
    def mean(self, axis=None, **kwargs):
        """
        Return the mean of the cube, optionally over an axis.
        """
        return self._compute(da.nanmean(self._get_filled_data(fill=np.nan), axis=axis, **kwargs))

    @projection_if_needed
    @ignore_warnings
    def median(self, axis=None, **kwargs):
        """
        Return the median of the cube, optionally over an axis.
        """
        data = self._get_filled_data(fill=np.nan)

        if axis is None:
            # da.nanmedian raises NotImplementedError since it is not possible
            # to do efficiently, so we use Numpy instead.
            self._warn_slow('median')
            return np.nanmedian(self._compute(data), **kwargs)
        else:
            return self._compute(da.nanmedian(self._get_filled_data(fill=np.nan), axis=axis, **kwargs))

    @projection_if_needed
    @ignore_warnings
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

        data = self._get_filled_data(fill=np.nan)

        if axis is None:
            # There is no way to compute the percentile of the whole array in
            # chunks.
            self._warn_slow('percentile')
            return np.nanpercentile(data, q, **kwargs)
        else:
            # Rechunk so that there is only one chunk along the desired axis
            data = data.rechunk([-1 if i == axis else 'auto' for i in range(3)])
            return self._compute(data.map_blocks(np.nanpercentile, q=q, drop_axis=axis, axis=axis, **kwargs))

    @projection_if_needed
    @ignore_warnings
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
        return self._compute(da.nanstd(self._get_filled_data(fill=np.nan), axis=axis, ddof=ddof, **kwargs))

    @projection_if_needed
    @ignore_warnings
    def mad_std(self, axis=None, ignore_nan=True, **kwargs):
        """
        Use astropy's mad_std to compute the standard deviation
        """

        data = self._get_filled_data(fill=np.nan)

        if axis is None:
            # In this case we have to load the full data - even dask's
            # nanmedian doesn't work efficiently over the whole array.
            self._warn_slow('mad_std')
            return stats.mad_std(data, ignore_nan=ignore_nan, **kwargs)
        else:
            # Rechunk so that there is only one chunk along the desired axis
            data = data.rechunk([-1 if i == axis else 'auto' for i in range(3)])
            return self._compute(data.map_blocks(stats.mad_std, drop_axis=axis,
                                                 axis=axis,
                                                 ignore_nan=ignore_nan,
                                                 **kwargs))

    @projection_if_needed
    @ignore_warnings
    def max(self, axis=None, **kwargs):
        """
        Return the maximum data value of the cube, optionally over an axis.
        """
        return self._compute(da.nanmax(self._get_filled_data(fill=np.nan), axis=axis, **kwargs))

    @projection_if_needed
    @ignore_warnings
    def min(self, axis=None, **kwargs):
        """
        Return the minimum data value of the cube, optionally over an axis.
        """
        return self._compute(da.nanmin(self._get_filled_data(fill=np.nan), axis=axis, **kwargs))

    @ignore_warnings
    def argmax(self, axis=None, **kwargs):
        """
        Return the index of the maximum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return self._compute(da.nanargmax(self._get_filled_data(fill=-np.inf), axis=axis, **kwargs))

    @ignore_warnings
    def argmin(self, axis=None, **kwargs):
        """
        Return the index of the minimum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return self._compute(da.nanargmin(self._get_filled_data(fill=np.inf), axis=axis))

    @ignore_warnings
    def statistics(self):
        """
        Return a dictinary of global basic statistics for the data.

        This method is designed to minimize the number of times each chunk is
        accessed. The statistics are computed for each chunk in turn before
        being aggregated.

        The names for each statistic are adopted from CASA's ia.statistics
        (see https://casa.nrao.edu/Release4.1.0/doc/CasaRef/image.statistics.html)
        """

        data = self._get_filled_data(fill=np.nan)

        def compute_stats(chunk, *args):
            # NOTE: we used to but do not use bottleneck here anymore, as it
            # does not seem to provide any performance improvements and also
            # has a memory leak when dealing with >f4 dtypes.
            return np.array([[[[np.sum(~np.isnan(chunk)),
                                np.nanmin(chunk),
                                np.nanmax(chunk),
                                np.nansum(chunk),
                                np.nansum(chunk * chunk)]]]])

        with dask.config.set(**self._scheduler_kwargs):
            results = da.map_blocks(compute_stats, data, new_axis=3).compute()

        count_values, min_values, max_values, sum_values, ssum_values = results.reshape((-1, 5)).T

        stats = {'npts': count_values.sum(),
                 'min': min_values.min() * self._unit,
                 'max': max_values.max() * self._unit,
                 'sum': sum_values.sum() * self._unit,
                 'sumsq': ssum_values.sum() * self._unit ** 2}

        stats['mean'] = stats['sum'] / stats['npts']

        # FIXME: for now this uses the simple 'textbook' algorithm which is not
        # numerically stable, so this should be replaced by a more robust approach
        stats['sigma'] = ((stats['sumsq'] - stats['sum'] ** 2 / stats['npts']) / (stats['npts'] - 1)) ** 0.5

        stats['rms'] = np.sqrt(stats['sumsq'] / stats['npts'])

        return stats

    def _map_blocks_to_cube(self, function, additional_arrays=None, fill=np.nan, rechunk=None,
                            return_new_cube=True,
                            **kwargs):
        """
        Call dask's map_blocks, returning a new spectral cube.
        """

        data = self._get_filled_data(fill=fill)

        if rechunk is not None:
            data = data.rechunk(rechunk)

        if additional_arrays is None:
            newdata = da.map_blocks(function, data, dtype=data.dtype, **kwargs)
        else:
            additional_arrays = [array.rechunk(data.chunksize) for array in additional_arrays]
            newdata = da.map_blocks(function, data, *additional_arrays, dtype=data.dtype, **kwargs)

        # Create final output cube
        if return_new_cube:
            newcube = self._new_cube_with(data=newdata,
                                          wcs=self.wcs,
                                          mask=self.mask,
                                          meta=self.meta,
                                          fill_value=self.fill_value)

            return newcube
        else:
            return newdata

    # NOTE: the following three methods could also be implemented spaxel by
    # spaxel using apply_function_parallel_spectral but then take longer (but
    # less memory)

    @add_save_to_tmp_dir_option
    def sigma_clip_spectrally(self,
                              threshold,
                              **kwargs):
        """
        Run astropy's sigma clipper along the spectral axis, converting all bad
        (excluded) values to NaN.

        Parameters
        ----------
        threshold : float
            The ``sigma`` parameter in `astropy.stats.sigma_clip`, which refers
            to the number of sigma above which to cut.
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
        kwargs : dict
            Passed to the sigma_clip function
        """

        def spectral_sigma_clip(array):
            return stats.sigma_clip(array, sigma=threshold, axis=0,
                                    masked=False, copy=True, **kwargs)

        return self.apply_function_parallel_spectral(spectral_sigma_clip,
                                                     accepts_chunks=True)

    @add_save_to_tmp_dir_option
    def spectral_smooth(self,
                        kernel,
                        convolve=convolution.convolve,
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
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
        kwargs : dict
            Passed to the convolve function
        """

        if isinstance(kernel.array, u.Quantity):
            raise u.UnitsError("The convolution kernel should be defined "
                               "without a unit.")

        def spectral_smooth(array):
            kernel_3d = kernel.array.reshape((len(kernel.array), 1, 1))
            return convolve(array, kernel_3d, normalize_kernel=True)

        return self.apply_function_parallel_spectral(spectral_smooth,
                                                     accepts_chunks=True)

    @add_save_to_tmp_dir_option
    def spectral_smooth_median(self, ksize, raise_error_jybm=True,
                               filter=ndimage.median_filter, **kwargs):
        return self.spectral_filter(ksize, filter=filter,
                                    raise_error_jybm=raise_error_jybm,
                                    **kwargs)

    @add_save_to_tmp_dir_option
    def spectral_filter(self, ksize, filter, raise_error_jybm=True,
                        **kwargs):
        """
        Smooth the cube along the spectral dimension using a scipy.ndimage filter.

        Parameters
        ----------
        ksize : int
            Size of the median filter in spectral channels (scipy.ndimage.median_filter).
        filter : function
            A filter from `scipy.ndimage <https://docs.scipy.org/doc/scipy/reference/ndimage.html#filters>`_.
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
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
            return filter(img, (ksize, 1, 1), **kwargs)

        return self.apply_function_parallel_spectral(median_filter_wrapper,
                                                     accepts_chunks=True)

    @add_save_to_tmp_dir_option
    def spatial_smooth(self, kernel, convolve=convolution.convolve, raise_error_jybm=True, **kwargs):
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
        raise_error_jybm : bool, optional
            Raises a `~spectral_cube.utils.BeamUnitsError` when smoothing a cube in Jy/beam units,
            since the brightness is dependent on the spatial resolution.
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
        kwargs : dict
            Passed to the convolve function
        """

        self.check_jybeam_smoothing(raise_error_jybm=raise_error_jybm)

        def convolve_wrapper(data, kernel=None, **kwargs):
            return convolve(data, kernel, normalize_kernel=True, **kwargs)

        return self.apply_function_parallel_spatial(convolve_wrapper, kernel=kernel.array)

    @add_save_to_tmp_dir_option
    def spatial_filter(self, ksize, filter, raise_error_jybm=True, **kwargs):
        """
        Smooth the image in each spatial-spatial plane of the cube using a median filter.

        Parameters
        ----------
        ksize : int
            Size of the filter in pixels.
        filter : function
            A filter from `scipy.ndimage <https://docs.scipy.org/doc/scipy/reference/ndimage.html#filters>`_.
        raise_error_jybm : bool, optional
            Raises a `~spectral_cube.utils.BeamUnitsError` when smoothing a cube in Jy/beam units,
            since the brightness is dependent on the spatial resolution.
        kwargs : dict
            Passed to the median_filter function
        """

        if not SCIPY_INSTALLED:
            raise ImportError("Scipy could not be imported: this function won't work.")

        self.check_jybeam_smoothing(raise_error_jybm=raise_error_jybm)

        def median_filter_wrapper(data, ksize=None, **kwargs):
            return filter(data, ksize, **kwargs)

        return self.apply_function_parallel_spatial(median_filter_wrapper, ksize=ksize)

    def spatial_smooth_median(self, ksize, raise_error_jybm=True,
            filter=ndimage.median_filter, **kwargs):
        """
        Smooth the image in each spatial-spatial plane of the cube using a median filter.
        """
        return self.spatial_filter(ksize=ksize, filter=filter,
                raise_error_jybm=raise_error_jybm, **kwargs)

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

        data = self._get_filled_data(fill=np.nan).astype(np.float64)
        pix_size = self._pix_size_slice(axis)
        pix_cen = self._pix_cen()[axis]

        if order == 0:
            out = nansum_allbadtonan(data * pix_size, axis=axis)
        else:
            denominator = self._compute(nansum_allbadtonan(data * pix_size, axis=axis))
            mom1 = (nansum_allbadtonan(data * pix_size * pix_cen, axis=axis) /
                    denominator)
            if order > 1:
                # insert an axis so it broadcasts properly
                shp = list(mom1.shape)
                shp.insert(axis, 1)
                mom1 = self._compute(mom1.reshape(shp))
                out = (nansum_allbadtonan(data * pix_size * (pix_cen - mom1) ** order, axis=axis) /
                       denominator)
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

        # We need to use a slightly different approach to SpectralCube here
        # because there isn't yet a dask-friendly version of find_objects
        # https://github.com/dask/dask-image/issues/96

        if isinstance(region_mask, np.ndarray):
            if is_broadcastable_and_smaller(region_mask.shape, self.shape):
                region_mask = BooleanArrayMask(region_mask, self._wcs)
            else:
                raise ValueError("Mask shape does not match cube shape.")

        include = region_mask.include(self._data, self._wcs,
                                      wcs_tolerance=self._wcs_tolerance)

        include = da.broadcast_to(include, self.shape)

        slices = []
        for axis in range(3):
            if axis == 0 and spatial_only:
                slices.append(slice(None))
                continue
            collapse_axes = tuple(index for index in range(3) if index != axis)
            valid = self._compute(da.any(include, axis=collapse_axes))
            if np.any(valid):
                indices = np.where(valid)[0]
                slices.append(slice(np.min(indices), np.max(indices) + 1))
            else:
                slices.append(slice(0))

        return tuple(slices)

    @add_save_to_tmp_dir_option
    def downsample_axis(self, factor, axis, estimator=np.nanmean,
                        truncate=False):
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
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
        """

        # FIXME: this does not work correctly currently due to
        # https://github.com/dask/dask/issues/6102

        warnings.warn('In some cases, the final shape of the output from downsample_axis '
                      'is incorrect, so use the result with caution', UserWarning)

        data = self._get_filled_data(fill=self._fill_value)
        mask = da.asarray(self.mask.include(), name=str(uuid.uuid4()))

        if not truncate and data.shape[axis] % factor != 0:
            padding_shape = list(data.shape)
            padding_shape[axis] = factor - data.shape[axis] % factor
            data_padding = da.ones(padding_shape) * np.nan
            mask_padding = da.zeros(padding_shape, dtype=bool)
            data = da.concatenate([data, data_padding], axis=axis)
            mask = da.concatenate([mask, mask_padding], axis=axis).rechunk()

        data = da.coarsen(estimator, data, {axis: factor}, trim_excess=True)
        mask = da.coarsen(estimator, mask, {axis: factor}, trim_excess=True)

        view = [slice(None, None, factor) if ii == axis else slice(None) for ii in range(self.ndim)]
        newwcs = wcs_utils.slice_wcs(self.wcs, view, shape=self.shape)
        newwcs._naxis = list(self.shape)

        # this is an assertion to ensure that the WCS produced is valid
        # (this is basically a regression test for #442)
        assert newwcs[:, slice(None), slice(None)]
        assert len(newwcs._naxis) == 3

        return self._new_cube_with(data=data, wcs=newwcs,
                                   mask=BooleanArrayMask(mask, wcs=newwcs))

    @add_save_to_tmp_dir_option
    def spectral_interpolate(self, spectral_grid,
                             suppress_smooth_warning=False,
                             fill_value=None,
                             force_rechunk=True):
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
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
        force_rechunk : bool
            If `True`, forces rechunking of the dask array to have a single chunk
            along the spectral axis. If `False`, the data will not be rechunked, but
            a ValueError is raised if rechunking is required to have a single chunk
            along the spectral axis.

        Returns
        -------
        cube : SpectralCube

        """

        # TODO: this duplicates SpectralCube.spectral_interpolate, so we should
        # find a way to avoid that duplication.

        inaxis = self.spectral_axis.to(spectral_grid.unit)

        indiff = np.mean(np.diff(inaxis))
        outdiff = np.mean(np.diff(spectral_grid))

        reverse_in = indiff < 0
        reverse_out = outdiff < 0

        # account for reversed axes

        if reverse_in:
            inaxis = inaxis[::-1]
            indiff = np.mean(np.diff(inaxis))

        if reverse_out:
            spectral_grid = spectral_grid[::-1]
            outdiff = np.mean(np.diff(spectral_grid))

        cubedata = self._get_filled_data(fill=np.nan)

        # insanity checks
        if indiff < 0 or outdiff < 0:
            raise ValueError("impossible.")

        assert np.all(np.diff(spectral_grid) > 0)
        assert np.all(np.diff(inaxis) > 0)

        np.testing.assert_allclose(np.diff(spectral_grid), outdiff,
                                   err_msg="Output grid must be linear")

        if outdiff > 2 * indiff and not suppress_smooth_warning:
            warnings.warn("Input grid has too small a spacing. The data should "
                          "be smoothed prior to resampling.", SmoothingWarning)

        if reverse_in:
            cubedata = cubedata[::-1, :, :]

        if force_rechunk:
            cubedata = cubedata.rechunk((-1, 'auto', 'auto'))
        else:
            # There should be one chunk size along the spectral
            # axis if there is only 1 chunk already defined.
            # Otherwise, the data needs to be rechunked.
            if len(cubedata.chunks[0]) > 1:
                raise ValueError(f"The cube currently has {len(cubedata.chunks[0])} chunks along"
                                 " the spectral axis but DaskSpectralCube.spectral_interpolate"
                                 " requires one. Rechunk the data first or enable"
                                 " `force_rechunk=True`.")

        chunkshape = (len(spectral_grid),) + cubedata.chunks[1:]

        def interp_wrapper(y, args):
            if y.size == 1:
                return y
            else:
                interp = scipy.interpolate.interp1d(args[1], y.T,
                                                    fill_value=fill_value,
                                                    bounds_error=False)
                return interp(args[0]).T

        newcube = cubedata.map_blocks(interp_wrapper,
                                      args=(spectral_grid.value, inaxis.value),
                                      chunks=chunkshape)

        newwcs = self.wcs.deepcopy()
        newwcs.wcs.crpix[2] = 1
        newwcs.wcs.crval[2] = spectral_grid[0].value if not reverse_out \
            else spectral_grid[-1].value
        newwcs.wcs.cunit[2] = spectral_grid.unit.to_string('FITS')
        newwcs.wcs.cdelt[2] = outdiff.value if not reverse_out \
            else -outdiff.value
        newwcs.wcs.set()

        newbmask = BooleanArrayMask(~np.isnan(newcube), wcs=newwcs)

        if reverse_out:
            newcube = newcube[::-1, :, :]

        newcube = self._new_cube_with(data=newcube, wcs=newwcs, mask=newbmask,
                                      meta=self.meta,
                                      fill_value=self.fill_value)

        return newcube


class DaskSpectralCube(DaskSpectralCubeMixin, SpectralCube):

    def __init__(self, data, *args, **kwargs):
        unit = None
        if not isinstance(data, da.Array):
            if isinstance(data, u.Quantity):
                data, unit = data.value, data.unit
            # NOTE: don't be tempted to chunk this image-wise (following the
            # data storage) because spectral operations will take forever.
            data = da.asarray(data, name=str(uuid.uuid4()))
        super().__init__(data, *args, **kwargs)
        if self._unit is None and unit is not None:
            self._unit = unit

    @classmethod
    def read(cls, *args, **kwargs):
        if kwargs.get('use_dask') is None:
            kwargs['use_dask'] = True
        return super().read(*args, **kwargs)

    def write(self, *args, **kwargs):
        with dask.config.set(**self._scheduler_kwargs):
            super().write(*args, **kwargs)

    @property
    def hdu(self):
        """
        HDU version of self
        """
        return PrimaryHDU(self._get_filled_data(fill=self._fill_value), header=self.header)

    @property
    def hdulist(self):
        return HDUList(self.hdu)

    @add_save_to_tmp_dir_option
    def convolve_to(self, beam, convolve=convolution.convolve, **kwargs):
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
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.
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

        if self.unit.is_equivalent(u.Jy / u.beam):
            beam_ratio_factor = (beam.sr / self.beam.sr).value
        else:
            beam_ratio_factor = 1.

        # See #631: kwargs get passed within self.apply_function_parallel_spatial
        def convfunc(img, **kwargs):
            return convolve(img, kernel, normalize_kernel=True, **kwargs).reshape(img.shape) * beam_ratio_factor

        if convolve is convolution.convolve_fft and 'allow_huge' not in kwargs:
            kwargs['allow_huge'] = self.allow_huge_operations

        return self.apply_function_parallel_spatial(convfunc,
                                                    accepts_chunks=True,
                                                    **kwargs).with_beam(beam, raise_error_jybm=False)


class DaskVaryingResolutionSpectralCube(DaskSpectralCubeMixin, VaryingResolutionSpectralCube):

    def __init__(self, data, *args, **kwargs):
        unit = None
        if not isinstance(data, da.Array):
            if isinstance(data, u.Quantity):
                data, unit = data.value, data.unit
            # NOTE: don't be tempted to chunk this image-wise (following the
            # data storage) because spectral operations will take forever.
            data = da.asarray(data, name=str(uuid.uuid4()))
        super().__init__(data, *args, **kwargs)
        if self._unit is None and unit is not None:
            self._unit = unit

    @classmethod
    def read(cls, *args, **kwargs):
        if kwargs.get('use_dask') is None:
            kwargs['use_dask'] = True
        return super().read(*args, **kwargs)

    def write(self, *args, **kwargs):
        with dask.config.set(**self._scheduler_kwargs):
            super().write(*args, **kwargs)

    @property
    def hdu(self):
        raise ValueError("For DaskVaryingResolutionSpectralCube's, use hdulist "
                         "instead of hdu.")

    @property
    def hdulist(self):
        """
        HDUList version of self
        """

        hdu = PrimaryHDU(self._get_filled_data(fill=self._fill_value), header=self.header)

        from .cube_utils import beams_to_bintable
        # use unmasked beams because, even if the beam is masked out, we should
        # write it
        bmhdu = beams_to_bintable(self.unmasked_beams)

        return HDUList([hdu, bmhdu])

    @add_save_to_tmp_dir_option
    def convolve_to(self, beam, allow_smaller=False,
                    convolve=convolution.convolve_fft,
                    **kwargs):
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
        save_to_tmp_dir : bool
            If `True`, the computation will be carried out straight away and
            saved to a temporary directory. This can improve performance,
            especially if carrying out several operations sequentially. If
            `False`, the computation is only carried out when accessing
            specific parts of the data or writing to disk.

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

        beams = []
        beam_ratio_factors = []
        for bm, valid in zip(self.unmasked_beams, self.goodbeams_mask):
            if not valid:
                # just skip masked-out beams
                beams.append(None)
                beam_ratio_factors.append(None)
                continue
            elif beam == bm:
                # Point response when beams are equal, don't convolve.
                beams.append(None)
                beam_ratio_factors.append(None)
                continue
            try:
                beams.append(beam.deconvolve(bm))
                beam_ratio_factors.append((beam.sr / bm.sr).value)
            except ValueError:
                if allow_smaller:
                    beams.append(None)
                    beam_ratio_factors.append(None)
                else:
                    raise

        # We need to pass in the beams to dask, so we hide them inside an object array
        # that can then be chunked like the data.
        beams = da.from_array(np.array(beams, dtype=object)
                              .reshape((len(beams), 1, 1)), chunks=(-1, -1, -1))

        needs_beam_ratio = self.unit.is_equivalent(u.Jy / u.beam)

        # See #631: kwargs get passed within self.apply_function_parallel_spatial
        def convfunc(img, beam, **kwargs):
            if img.size > 0:
                out = np.zeros(img.shape, dtype=img.dtype)
                for index in range(img.shape[0]):
                    if beam[index, 0, 0] is None:
                        out[index] = img[index]
                    else:
                        kernel = beam[index, 0, 0].as_kernel(pixscale)
                        out[index] = convolve(img[index], kernel, normalize_kernel=True, **kwargs)

                        if needs_beam_ratio and beam_ratio_factors[index] is not None:
                            out[index] *= beam_ratio_factors[index]

                return out
            else:
                return img

        # Rechunk so that there is only one chunk in the image plane
        cube = self._map_blocks_to_cube(convfunc,
                                        additional_arrays=(beams,),
                                        rechunk=('auto', -1, -1),
                                        **kwargs)

        # Result above is a DaskVaryingResolutionSpectralCube, convert to DaskSpectralCube
        newcube = DaskSpectralCube(data=cube._data,
                                   beam=beam,
                                   wcs=cube.wcs,
                                   mask=cube.mask,
                                   meta=cube.meta,
                                   fill_value=cube.fill_value)

        newcube._scheduler_kwargs = self._scheduler_kwargs

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

    @property
    def _mask_include(self):
        return BooleanArrayMask(da.from_array(MaskHandler(self),
                                              name='MaskHandler ' + str(uuid.uuid4()),
                                              chunks=self._data.chunksize),
                                wcs=self.wcs,
                                shape=self.shape)
