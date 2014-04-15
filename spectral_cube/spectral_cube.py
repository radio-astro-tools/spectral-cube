"""
A class to represent a 3-d position-position-velocity spectral cube.
"""

from abc import ABCMeta, abstractproperty

from astropy import units as u
import numpy as np

from . import cube_utils

__all__ = ['SpectralCube']


class MaskBase(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def include(self):
        pass

    def _flattened(self, cube, slices):
        """
        Return a flattened array of the included elements of cube

        Parameters
        ----------
        cube : array-like
           The cube to extract

        Returns
        -------
        A 1D ndarray

        Notes
        -----
        This is an internal method used by :class:`SpectralCube`.
        """
        if slices is None:
            return cube[self.include]
        else:
            return cube[slices][self.include[slices]]

    def _filled(self, array, fill=np.nan):
        """
        Replace the exluded elements of *array* with *fill*.

        Parameters
        ----------
        array : array-like
            Input array
        fill : number
            Replacement value

        Returns
        -------
        A new array

        Notes
        -----
        This is an internal method used by :class:`SpectralCube`.
        Users should use :meth:`SpectralCubeMask.get_data`
        """
        return np.where(self.include, array, fill)

    @property
    def exclude(self):
        return np.logical_not(self.include)


class SpectralCubeMask(MaskBase):

    def __init__(self, mask, wcs, include=True):
        mask, self._wcs = cube_utils._orient(mask, wcs)
        self._includemask = mask if include else np.logical_not(mask)
        
    def __repr__(self):
        return "SpectralCubeMask with shape {0}: {1}".format(str(self.shape),
                                                             self._includemask.__repr__())

    @property
    def include(self):
        return self._includemask

    def _include(self, slice):
        # this is what gets overridden
        return self._includemask[slice]

    @property
    def shape(self):
        return self._includemask.shape

    # use subcube instead
    #def __getitem__(self, slice):
    #    # TODO: need to update WCS!
    #    return SpectralCubeMask(self._includemask[slice], self._wcs)


class SpectralCube(object):

    def __init__(self, data, wcs, mask=None, meta=None):
        # TODO: mask should be oriented? Or should we assume correctly oriented here?
        self._data, self._wcs = cube_utils._orient(data, wcs)
        self._spectral_axis = None
        self._mask = mask  # specifies which elements to Nan/blank/ignore -> SpectralCubeMask
                           # object or array-like object, given that WCS needs to be consistent with data?
        #assert mask._wcs == self._wcs
        self.meta = meta or {}

    @property
    def shape(self):
        return self._data.shape

    # This should just be relegated to subcube
    #def __getitem__(self, slice):
    #    # TODO: need to update WCS!
    #    return SpectralCube(self._data[slice], self._wcs,
    #                        mask=self._mask[slice], meta=self.meta)

    def __repr__(self):
        return "SpectralCube with shape {0}: {1}".format(str(self.shape),
                                                         self._data.__repr__())

    @classmethod
    def read(cls, filename, format=None):
        if format == 'fits':
            from .io.fits import load_fits_cube
            return load_fits_cube(filename)
        elif format == 'casa_image':
            from .io.casa_image import load_casa_image
            return load_casa_image(filename)
        else:
            raise ValueError("Format {0} not implemented".format(format))

    def write(self, filename, format=None, includestokes=False, clobber=False):
        if format == 'fits':
            write_fits(filename, self._data, self._wcs,
                       includestokes=includestokes, clobber=clobber)
        else:
            raise NotImplementedError("Try FITS instead")

    def _apply_numpy_function(self, function, fill=np.nan, **kwargs):
        """
        Apply a numpy function to the cube
        """
        return function(self.get_data(fill=fill), **kwargs)

    def sum(self, axis=None):
        # use nansum, and multiply by mask to add zero each time there is badness
        return self._apply_numpy_function(np.nansum, fill=np.nan, axis=axis)

    def max(self, axis=None):
        return self._apply_numpy_function(np.nanmax, fill=np.nan, axis=axis)

    def min(self, axis=None):
        return self._apply_numpy_function(np.nanmin, fill=np.nan, axis=axis)

    def argmax(self, axis=None):
        return self._apply_numpy_function(np.nanargmax, fill=np.nan, axis=axis)

    def argmin(self, axis=None):
        return self._apply_numpy_function(np.nanargmin, fill=np.nan, axis=axis)

    @property
    def data_valid(self):
        """ Flat array of unmasked data values """
        return self._data[self.mask.include]

    def chunked(self, chunksize=1000):
        """
        Iterate over chunks of valid data
        """
        raise NotImplementedError()

    def _get_flat_shape(self, axis):
        """
        Get the shape of the array after flattening along an axis
        """
        iteraxes = [0,1,2]
        iteraxes.remove(axis)
        # x,y are defined as first,second dim to iterate over
        # (not x,y in pixel space...)
        nx = self.shape[iteraxes[0]]
        ny = self.shape[iteraxes[1]]
        return nx,ny

    def _apply_along_axes(self, function, axis=None, weights=None, **kwargs):
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
        
        # determine the output array shape
        nx,ny = self._get_flat_shape(axis)

        # allocate memory for output array
        out = np.empty([nx,ny])

        # iterate over "lines of sight" through the cube
        for x,y,slc in self._iter_rays(axis):
            # acquire the flattened, valid data for the slice
            data = self.flattened(slc, weights=weights)
            # store result in array
            out[x,y] = function(data, **kwargs)

        return out

    def _iter_rays(self, axis=None):
        """
        Iterate over slices corresponding to lines-of-sight through a cube
        along the specified axis
        """
        nx,ny = self._get_flat_shape(axis)

        for x in xrange(nx):
            for y in xrange(ny):
                # create length-1 slices for each position
                slc = [slice(x,x+1),slice(y,y+1)]
                # create a length-N slice (all-inclusive) along the selected axis
                slc.insert(axis,slice(None))
                yield x,y,slc

    def flattened(self, slice=None, weights=None):
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
        data = self._mask._flattened(self._data, slice)
        if weights is not None:
            weights = self._mask._flattened(weights, slice)
            return data*weights
        else:
            return data
        
    def median(self, axis=None):
        return self._apply_along_axes(np.median, axis=axis)

    def percentile(self, q, axis=None):
        return self._apply_along_axes(np.percentile, q=q, axis=axis)

    # probably do not want to support this
    #def get_masked_array(self):
    #    return np.ma.masked_where(self.mask, self._data)

    def get_data(self, fill=np.nan):
        """
        Return the underlying data as a numpy array.
        Always returns the spectral axis as the 0th axis

        Sets masked values to *fill*
        """
        if self._mask is None:
            return self._data

        return self._mask._filled(self._data, fill)

    @property
    def data_unmasked(self):
        """
        Like data, but don't apply the mask
        """
        return self._data

    def apply_mask(self, mask, inherit=True):
        """
        Return a new Cube object, that applies the input mask to the underlying data.

        Handles any necessary logic to resample the mask

        If inherit=True, the new Cube uses the union of the original and new mask

        What type of object is `mask`? SpectralMask?
        Not sure -- I guess it needs to have WCS. Conceptually maps onto a boolean ndarray
            CubeMask? -> maybe SpectralCubeMask to be consistent with SpectralCube

        """

    def moment(self, order, axis, wcs=False):
        """
        Determine the n'th moment along the spectral axis

        If *wcs = True*, return the WCS describing the moment
        """

        nx,ny = self._get_flat_shape(axis)

        # allocate memory for output array
        out = np.empty([nx,ny])

        for x,y,slc in self._iter_rays(axis):
            # compute the world coordinates along the specified axis
            coords = self.world(axis, slc)
            # the numerator of the moment sum
            data = self.flattened(slc, weights=coords**order)
            # the denominator of the moment sum
            weights = self.flattened(slc)

            out[x,y] = data.sum()/weights.sum()

        return out

    @property
    def spectral_axis(self):
        """
        A `~astropy.units.Quantity` array containing the central values of
        each channel along the spectral axis.
        """

        # TODO: use world[...] once implemented
        iz, iy, ix = np.broadcast_arrays(np.arange(self.shape[0]), 0., 0.)
        return self._wcs.all_pix2world(ix, iy, iz, 0)[2] * u.Unit(self._wcs.wcs.cunit[2])

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
        elif ilo == ihi:
            ihi += 1

        # Create WCS slab
        wcs_slab = self._wcs.copy()
        wcs_slab.wcs.crpix[2] -= ilo

        # Create mask slab
        if self._mask is None:
            mask_slab = None
        else:
            mask_slab = self._mask[ilo:ihi]

        # Create new spectral cube
        slab = SpectralCube(self._data[ilo:ihi], wcs_slab,
                            mask=mask_slab, meta=self.meta)

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

    def world(self, view):

        inds = np.ogrid[[slice(0, s) for s in self._data.shape]]
        inds = np.broadcast_arrays(*inds)
        inds = [i[view] for i in inds[::-1]]  # numpy -> wcs order

        shp = inds[0].shape
        inds = np.column_stack([i.ravel() for i in inds])
        world = self._wcs.all_pix2world(inds, 0).T

        world = [w.reshape(shp) for w in world]  # 1D->3D
        return world[::-1]  # reverse WCS -> numpy order


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

def write(cube, filename, format=None, include_stokes=False, clobber=False):
    if format == 'fits':
        from .io.fits import write_fits_cube
        write_fits_cube(filename, cube,
                        include_stokes=include_stokes, clobber=clobber)
    else:
        raise NotImplementedError("Try FITS instead")


def test():
    x = SpectralCube()
    x2 = SpectralCube()
    x.sum(axis='spectral')
    # optionally:
    x.sum(axis=0)  # where 0 is defined to be spectral
    x.moment(3)  # kurtosis?  moment assumes spectral
    (x * x2).sum(axis='spatial1')
