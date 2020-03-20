"""
A class to represent a 3-d position-position-velocity spectral cube.
"""

from __future__ import print_function, absolute_import, division

import warnings

from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_area

import numpy as np
import dask.array as da

from astropy import stats
from astropy import convolution

from . import wcs_utils
from .spectral_cube import SpectralCube, SIGMA2FWHM, np2wcs
from .utils import warn_slow, VarianceWarning
from .lower_dimensional_structures import Projection

__all__ = ['DaskSpectralCube']


class DaskSpectralCube(SpectralCube):

    @property
    def _nan_filled_dask_array(self):
        return da.asarray(self._mask._filled(data=self._data,
                                             wcs=self._wcs, fill=np.nan,
                                             wcs_tolerance=self._wcs_tolerance))

    @warn_slow
    def sum(self, axis=None, **kwargs):
        """
        Return the sum of the cube, optionally over an axis.
        """
        return da.nansum(self._nan_filled_dask_array, axis=axis).compute()

    @warn_slow
    def mean(self, axis=None, **kwargs):
        """
        Return the mean of the cube, optionally over an axis.
        """
        return da.nanmean(self._nan_filled_dask_array, axis=axis).compute()

    @warn_slow
    def median(self, axis=None, **kwargs):
        """
        Return the median of the cube, optionally over an axis.
        """
        return da.nanmedian(self._nan_filled_dask_array, axis=axis).compute()

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
        return da.nanpercentile(self._nan_filled_dask_array, q, axis=axis).compute()

    @warn_slow
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
        return da.nanstd(self._nan_filled_dask_array, axis=axis, ddof=ddof).compute()

    @warn_slow
    def mad_std(self, axis=None, **kwargs):
        """
        Use astropy's mad_std to compute the standard deviation
        """

        data = self._nan_filled_dask_array

        if axis is None:
            # In this case we have to load the full data - even dask's
            # nanmedian doesn't work efficiently over the whole array.
            return stats.mad_std(data)
        else:
            # Rechunk so that there is only one chunk along the desired axis
            data = data.rechunk([-1 if i == axis else 'auto' for i in range(3)])
            return data.map_blocks(stats.mad_std, drop_axis=axis, axis=axis).compute()

    @warn_slow
    def max(self, axis=None, **kwargs):
        """
        Return the maximum data value of the cube, optionally over an axis.
        """
        return da.nanmax(self._nan_filled_dask_array, axis=axis).compute()

    @warn_slow
    def min(self, axis=None, **kwargs):
        """
        Return the minimum data value of the cube, optionally over an axis.
        """
        return da.nanmin(self._nan_filled_dask_array, axis=axis).compute()

    @warn_slow
    def argmax(self, axis=None, **kwargs):
        """
        Return the index of the maximum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return da.nanargmax(self._nan_filled_dask_array, axis=axis).compute()

    @warn_slow
    def argmin(self, axis=None, **kwargs):
        """
        Return the index of the minimum data value.

        The return value is arbitrary if all pixels along ``axis`` are
        excluded from the mask.
        """
        return da.nanargmin(self._nan_filled_dask_array, axis=axis).compute()

    def _map_blocks_to_cube(self, function, rechunk=None):
        """
        Call dask's map_blocks, returning a new spectral cube.
        """

        if rechunk is None:
            data = self._nan_filled_dask_array
        else:
            data = self._nan_filled_dask_array.rechunk(rechunk)

        newdata = data.map_blocks(function)

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

        # Rechunk so that there is only one chunk spectrally and let dask
        # decide for the rest
        return self._map_blocks_to_cube(spectral_smooth,
                                        rechunk=(-1, 'auto', 'auto'))

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

        convolution_kernel = beam.as_kernel(pixscale)
        kernel = convolution_kernel.array.reshape((1,) + convolution_kernel.array.shape)

        def convfunc(img):
            if img.size > 0:
                return convolve(img, kernel, normalize_kernel=True, **kwargs).reshape(img.shape)
            else:
                return img

        # Rechunk so that there is only one chunk in the image plane and let
        # dask decide for the rest
        return self._map_blocks_to_cube(convfunc,
                                        rechunk=('auto', -1, -1))

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

        data = self._nan_filled_dask_array

        pix_size = self._pix_size_slice(axis)
        pix_cen = self._pix_cen()[axis]

        if order == 0:
            out = da.nansum(data * pix_size, axis=axis)
        else:
            mom1 = (da.nansum(data * pix_size * pix_cen, axis=axis) /
                    da.nansum(data * pix_size, axis=axis))
            if order > 1:
                out = (da.nansum(data * pix_size * (pix_cen - mom1) ** order, axis=axis) /
                       da.nansum(data * pix_size, axis=axis))
            else:
                out = mom1

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
