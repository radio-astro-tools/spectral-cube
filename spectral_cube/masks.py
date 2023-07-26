from __future__ import print_function, absolute_import, division

import abc
import uuid
import warnings
import tempfile

from six.moves import zip
import numpy as np
from numpy.lib.stride_tricks import as_strided

import dask.array as da

from astropy.wcs import InconsistentAxisTypesError
from astropy.io import fits

from . import wcs_utils
from .utils import WCSWarning


__all__ = ['MaskBase', 'InvertedMask', 'CompositeMask', 'BooleanArrayMask',
           'LazyMask', 'LazyComparisonMask', 'FunctionMask']

# Global version of the with_spectral_unit docs to avoid duplicating them
with_spectral_unit_docs = """
        Parameters
        ----------
        unit : u.Unit
            Any valid spectral unit: velocity, (wave)length, or frequency.
            Only vacuum units are supported.
        velocity_convention : u.doppler_relativistic, u.doppler_radio, or u.doppler_optical
            The velocity convention to use for the output velocity axis.
            Required if the output type is velocity.
        rest_value : u.Quantity
            A rest wavelength or frequency with appropriate units.  Required if
            output type is velocity.  The cube's WCS should include this
            already if the *input* type is velocity, but the WCS's rest
            wavelength/frequency can be overridden with this parameter.
        """

def is_broadcastable_and_smaller(shp1, shp2):
    """
    Test if shape 1 can be broadcast to shape 2, not allowing the case
    where shape 2 has a dimension length 1
    """
    for a, b in zip(shp1[::-1], shp2[::-1]):
        # b==1 is broadcastable but not desired
        if a == 1 or a == b:
            pass
        else:
            return False
    return True

def dims_to_skip(shp1, shp2):
    """
    For a shape `shp1` that is broadcastable to shape `shp2`, specify which
    dimensions are length 1.

    Parameters
    ----------
    keep : bool
        If True, return the dimensions to keep rather than those to remove
    """
    if not is_broadcastable_and_smaller(shp1, shp2):
        raise ValueError("Cannot broadcast {0} to {1}".format(shp1,shp2))
    dims = []

    for ii,(a, b) in enumerate(zip(shp1[::-1], shp2[::-1])):
        # b==1 is broadcastable but not desired
        if a == 1:
            dims.append(len(shp2) - ii - 1)
        elif a == b:
            pass
        else:
            raise ValueError("This should not be possible")

    if len(shp1) < len(shp2):
        dims += list(range(len(shp2)-len(shp1)))

    return dims

def view_of_subset(shp1, shp2, view):
    """
    Given two shapes and a view, assuming that shape 1 can be broadcast
    to shape 2, return the sub-view that applies to shape 1
    """
    # if the view is 1-dimensional, we can't subset it
    if not hasattr(view, '__len__'):
        return view

    dts = dims_to_skip(shp1, shp2)
    if view:
        cv_view = [x for ii,x in enumerate(view) if ii not in dts]
    else:
        # if no view is specified, still need to slice
        cv_view = [x for ii,x in enumerate([slice(None)]*3)
                   if ii not in dts]

    # return type matters
    # array[[0,0]] = [array[0], array[0]]
    # array[(0,0)] = array[0,0]
    return tuple(cv_view)

class MaskBase(object):

    __metaclass__ = abc.ABCMeta

    def include(self, data=None, wcs=None, view=(), **kwargs):
        """
        Return a boolean array indicating which values should be included.

        If ``view`` is passed, only the sliced mask will be returned, which
        avoids having to load the whole mask in memory. Otherwise, the whole
        mask is returned in-memory.

        kwargs are passed to _validate_wcs
        """
        self._validate_wcs(data, wcs, **kwargs)
        return self._include(data=data, wcs=wcs, view=view)

    # Commented out, but left as a possibility, because including this didn't fix any
    # of the problems we encountered with matplotlib plotting
    def view(self, view=()):
        """
        Compatibility tool: if a numpy.ma.ufunc is run on the mask, it will try
        to grab a view of the mask, which needs to appear to numpy as a true
        array.  This can be important for, e.g., plotting.

        Numpy's convention is that masked=True means "masked out"

        .. note::
            I don't know if there are broader concerns or consequences from
            including this 'view' tool here.
        """
        return self.exclude(view=view)

    def _validate_wcs(self, new_data=None, new_wcs=None, **kwargs):
        """
        This method can be overridden in cases where the data and WCS have to
        conform to some rules. This gets called automatically when
        ``include`` or ``exclude`` are called.
        """
        pass

    @abc.abstractmethod
    def _include(self, data=None, wcs=None, view=()):
        pass

    def exclude(self, data=None, wcs=None, view=(), **kwargs):
        """
        Return a boolean array indicating which values should be excluded.

        If ``view`` is passed, only the sliced mask will be returned, which
        avoids having to load the whole mask in memory. Otherwise, the whole
        mask is returned in-memory.

        kwargs are passed to _validate_wcs
        """
        self._validate_wcs(data, wcs, **kwargs)
        return self._exclude(data=data, wcs=wcs, view=view)

    def _exclude(self, data=None, wcs=None, view=()):
        return np.logical_not(self._include(data=data, wcs=wcs, view=view))

    def any(self):
        return np.any(self.exclude())

    def _flattened(self, data, wcs=None, view=()):
        """
        Return a flattened array of the included elements of cube

        Parameters
        ----------
        data : array-like
            The data array to flatten
        view : tuple, optional
            Any slicing to apply to the data before flattening

        Returns
        -------
        flat_array : `~numpy.ndarray`
            A 1-D ndarray containing the flattened output

        Notes
        -----
        This is an internal method used by :class:`SpectralCube`.
        """

        mask = self.include(data=data, wcs=wcs, view=view)

        # Workaround for https://github.com/dask/dask/issues/6089
        if isinstance(data, da.Array) and not isinstance(mask, da.Array):
            mask = da.asarray(mask, name=str(uuid.uuid4()))

        # if not isinstance(data, da.Array) and isinstance(mask, da.Array):
        #     mask = mask.compute()

        return data[view][mask]

    def _filled(self, data, wcs=None, fill=np.nan, view=(), use_memmap=False,
                **kwargs):
        """
        Replace the excluded elements of *array* with *fill*.

        Parameters
        ----------
        data : array-like
            Input array
        fill : number
            Replacement value
        view : tuple, optional
            Any slicing to apply to the data before flattening
        use_memmap : bool
            Use a memory map to store the output data?

        Returns
        -------
        filled_array : `~numpy.ndarray`
            A 1-D ndarray containing the filled output

        Notes
        -----
        This is an internal method used by :class:`SpectralCube`.
        Users should use the property :meth:`MaskBase.filled_data`
        """
        # Must convert to floating point, but should not change from inherited
        # type otherwise
        dt = np.result_type(data.dtype, 0.0)

        if use_memmap and data.size > 0:
            ntf = tempfile.NamedTemporaryFile()
            sliced_data = np.memmap(ntf, mode='w+', shape=data[view].shape,
                                    dtype=dt)
            sliced_data[:] = data[view]
        else:
            sliced_data = data[view].astype(dt)

        ex = self.exclude(data=data, wcs=wcs, view=view, **kwargs)

        return np.ma.masked_array(sliced_data, mask=ex).filled(fill)

    def __and__(self, other):
        return CompositeMask(self, other, operation='and')

    def __or__(self, other):
        return CompositeMask(self, other, operation='or')

    def __xor__(self, other):
        return CompositeMask(self, other, operation='xor')

    def __invert__(self):
        return InvertedMask(self)

    @property
    def shape(self):
        raise NotImplementedError("{0} mask classes do not have shape attributes."
                                  .format(self.__class__.__name__))

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def dtype(self):
        return np.dtype('bool')

    def __getitem__(self):
        raise NotImplementedError("Slicing not supported by mask class {0}"
                                  .format(self.__class__.__name__))

    def quicklook(self, view, wcs=None, filename=None, use_aplpy=True,
                  aplpy_kwargs={}):
        '''
        View a 2D slice of the mask, specified by view.

        Parameters
        ----------
        view : tuple
            Slicing to apply to the mask. Must return a 2D slice.
        wcs : astropy.wcs.WCS, optional
            WCS object to use in plotting the mask slice.
        filename : str, optional
            Filename of the output image. Enables saving of the plot.
        use_aplpy : bool, optional
            Try plotting with the aplpy package
        aplpy_kwargs : dict, optional
            kwargs passed to `~aplpy.FITSFigure`.
        '''

        view_twod = self.include(view=view, wcs=wcs)

        if use_aplpy:

            if wcs is not None:
                hdu = fits.PrimaryHDU(view_twod.astype(int), wcs.to_header())
            else:
                hdu = fits.PrimaryHDU(view_twod.astype(int))

            try:
                import aplpy
                FITSFigure = aplpy.FITSFigure(hdu,
                                              **aplpy_kwargs)

                FITSFigure.show_grayscale()
                FITSFigure.add_colorbar()
                if filename is not None:
                    FITSFigure.save(filename)
            except (InconsistentAxisTypesError, ImportError):
                use_aplpy = True

        if not use_aplpy:
            from matplotlib import pyplot
            figure = pyplot.imshow(view_twod)
            if filename is not None:
                figure.savefig(filename)

    def _get_new_wcs(self, unit, velocity_convention=None, rest_value=None):
        """
        Returns a new WCS with a different Spectral Axis unit
        """
        from .spectral_axis import convert_spectral_axis,determine_ctype_from_vconv

        out_ctype = determine_ctype_from_vconv(self._wcs.wcs.ctype[self._wcs.wcs.spec],
                                               unit,
                                               velocity_convention=velocity_convention)

        newwcs = convert_spectral_axis(self._wcs, unit, out_ctype,
                                       rest_value=rest_value)
        newwcs.wcs.set()

        return newwcs

    _get_new_wcs.__doc__ += with_spectral_unit_docs


class InvertedMask(MaskBase):

    def __init__(self, mask):
        self._mask = mask

    @property
    def shape(self):
        return self._mask.shape

    def _include(self, data=None, wcs=None, view=()):
        return np.logical_not(self._mask.include(data=data, wcs=wcs, view=view))

    def __getitem__(self, view):
        return InvertedMask(self._mask[view])

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        """
        Get an InvertedMask copy with a WCS in the modified unit
        """
        newmask = self._mask.with_spectral_unit(unit,
                                                velocity_convention=velocity_convention,
                                                rest_value=rest_value)
        return InvertedMask(newmask)

    with_spectral_unit.__doc__ += with_spectral_unit_docs


class CompositeMask(MaskBase):
    """
    A combination of several masks.  The included masks are treated with the specified
    operation.

    Parameters
    ----------
    mask1, mask2 : Masks
        The two masks to composite
    operation : str
        Either 'and' or 'or'; the operation used to combine the masks
    """

    def __init__(self, mask1, mask2, operation='and'):
        if isinstance(mask1, np.ndarray) and isinstance(mask2, MaskBase) and hasattr(mask2, 'shape'):
            if not is_broadcastable_and_smaller(mask1.shape, mask2.shape):
                raise ValueError("Mask1 shape is not broadcastable to Mask2 shape: "
                                 "%s vs %s" % (mask1.shape, mask2.shape))
            mask1 = BooleanArrayMask(mask1, mask2._wcs, shape=mask2.shape)
        elif isinstance(mask2, np.ndarray) and isinstance(mask1, MaskBase) and hasattr(mask1, 'shape'):
            if not is_broadcastable_and_smaller(mask2.shape, mask1.shape):
                raise ValueError("Mask2 shape is not broadcastable to Mask1 shape: "
                                 "%s vs %s" % (mask2.shape, mask1.shape))
            mask2 = BooleanArrayMask(mask2, mask1._wcs, shape=mask1.shape)

        # both entries must have compatible, which effectively means
        # equal, WCSes.  Unless one is a function.
        if hasattr(mask1, '_wcs') and hasattr(mask2, '_wcs'):
            mask1._validate_wcs(new_data=None, wcs=mask2._wcs)
            # In order to composite composites, they must have a _wcs defined.
            # (maybe this should be a property?)
            self._wcs = mask1._wcs
        elif hasattr(mask1, '_wcs'):
            # if one mask doesn't have a WCS, but the other does, the
            # compositemask should have the same WCS as the one that does
            self._wcs = mask1._wcs
        elif hasattr(mask2, '_wcs'):
            self._wcs = mask2._wcs

        self._mask1 = mask1
        self._mask2 = mask2
        self._operation = operation

    def _validate_wcs(self, new_data=None, new_wcs=None, **kwargs):
        self._mask1._validate_wcs(new_data=new_data, new_wcs=new_wcs, **kwargs)
        self._mask2._validate_wcs(new_data=new_data, new_wcs=new_wcs, **kwargs)

    @property
    def shape(self):
        try:
            assert self._mask1.shape == self._mask2.shape
            return self._mask1.shape
        except AssertionError:
            raise ValueError("The composite mask does not have a well-defined "
                             "shape; its two components have shapes {0} and "
                             "{1}.".format(self._mask1.shape,
                                           self._mask2.shape))
        except NotImplementedError:
            raise ValueError("The composite mask contains at least one "
                             "component with no defined shape.")

    def _include(self, data=None, wcs=None, view=()):
        result_mask_1 = self._mask1._include(data=data, wcs=wcs, view=view)
        result_mask_2 = self._mask2._include(data=data, wcs=wcs, view=view)
        if self._operation == 'and':
            return np.bitwise_and(result_mask_1, result_mask_2)
        elif self._operation == 'or':
            return np.bitwise_or(result_mask_1, result_mask_2)
        elif self._operation == 'xor':
            return np.bitwise_xor(result_mask_1, result_mask_2)
        else:
            raise ValueError("Operation '{0}' not supported".format(self._operation))

    def __getitem__(self, view):
        return CompositeMask(self._mask1[view], self._mask2[view],
                             operation=self._operation)

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        """
        Get a CompositeMask copy in which each component has a WCS in the
        modified unit
        """
        newmask1 = self._mask1.with_spectral_unit(unit,
                                                  velocity_convention=velocity_convention,
                                                  rest_value=rest_value)
        newmask2 = self._mask2.with_spectral_unit(unit,
                                                  velocity_convention=velocity_convention,
                                                  rest_value=rest_value)
        return CompositeMask(newmask1, newmask2, self._operation)

    with_spectral_unit.__doc__ += with_spectral_unit_docs


class BooleanArrayMask(MaskBase):

    """
    A mask defined as an array on a spectral cube WCS

    Parameters
    ----------
    mask: `numpy.ndarray`
        A boolean numpy ndarray
    wcs: `astropy.wcs.WCS`
        The WCS object
    shape: tuple
        The shape of the region the array is masking.  This is *required* if
        ``mask.ndim != data.ndim`` to provide rules for how to broadcast the
        mask
    """

    def __init__(self, mask, wcs, shape=None, include=True):
        self._mask_type = 'include' if include else 'exclude'
        self._wcs = wcs
        self._wcs_whitelist = set()
        #if mask.ndim != 3 and (shape is None or len(shape) != 3):
        #    raise ValueError("When creating a BooleanArrayMask with <3 dimensions, "
        #                     "the shape of the 3D array must be specified.")
        if shape is not None and not is_broadcastable_and_smaller(mask.shape, shape):
            raise ValueError("Mask cannot be broadcast to the specified shape.")
        self._shape = shape or mask.shape
        self._mask = mask

        """
        Developer note (AG):
            The logic in this following section seems overly complicated.  All
            of it is added to make sure that a 1D boolean array along the
            spectral axis can be created.  I thought this was possible
            previously, but experience many errors in my latest attempt to use
            one.
        """
        # If a shape is given, we may need to broadcast to that shape
        if shape is not None:
            # these are dimensions that simply don't exist
            n_empty_dims = (len(self._shape)-mask.ndim)

            # these are dimensions of shape 1 that would be squeezed away but may
            # be needed to make the arrays broadcastable (e.g., mask[:,None,None])
            # Need to add n_empty_dims because (1,2) will broadcast to (3,1,2)
            # and there will be no extra dims.
            extra_dims = [ii
                          for ii,(sh1,sh2) in
                          enumerate(zip((0,)*n_empty_dims + mask.shape, shape))
                          if sh1 == 1 and sh1 != sh2]


            # Add the [None,]'s and the nonexistant
            n_extra_dims =  n_empty_dims + len(extra_dims)

            # if there are no extra dims, we're done, the original shape is fine
            if n_extra_dims > 0:
                strides = (0,)*n_empty_dims + mask.strides

                for ed in extra_dims:
                    # all of the [None,] dims should have 0 stride
                    assert strides[ed] == 0,"Stride shape failure"

                self._mask = as_strided(mask, shape=self.shape,
                                        strides=strides)

        # Make sure the mask shape matches the Mask object shape
        assert self._mask.shape == self.shape,"Shape initialization failure"

    def _validate_wcs(self, new_data=None, new_wcs=None, **kwargs):
        """
        Check that the new WCS matches the current one

        Parameters
        ----------
        kwargs : dict
            Passed to `wcs_utils.check_equality`
        """
        if new_data is not None and not is_broadcastable_and_smaller(self._mask.shape,
                                                                     new_data.shape):
            raise ValueError("data shape cannot be broadcast to match mask shape")
        if new_wcs is not None:
            if new_wcs not in self._wcs_whitelist:
                try:
                    if not wcs_utils.check_equality(new_wcs, self._wcs,
                                                    warn_missing=True,
                                                    **kwargs):
                        raise ValueError("WCS does not match mask WCS")
                    else:
                        self._wcs_whitelist.add(new_wcs)
                except InconsistentAxisTypesError:
                    warnings.warn("Inconsistent axis type encountered; WCS is "
                                  "invalid and therefore will not be checked "
                                  "against other WCSes.",
                                  WCSWarning
                                  )
                    self._wcs_whitelist.add(new_wcs)

    def _include(self, data=None, wcs=None, view=()):
        result_mask = self._mask[view]
        return result_mask if self._mask_type == 'include' else np.logical_not(result_mask)

    def _exclude(self, data=None, wcs=None, view=()):
        result_mask = self._mask[view]
        return result_mask if self._mask_type == 'exclude' else np.logical_not(result_mask)

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, view):
        return BooleanArrayMask(self._mask[view],
                                wcs_utils.slice_wcs(self._wcs, view,
                                                    shape=self.shape,
                                                    drop_degenerate=True),
                                shape=self._mask[view].shape)

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        """
        Get a BooleanArrayMask copy with a WCS in the modified unit
        """
        newwcs = self._get_new_wcs(unit, velocity_convention, rest_value)

        newmask = BooleanArrayMask(self._mask, newwcs,
                                   include=self._mask_type=='include')
        return newmask

    with_spectral_unit.__doc__ += with_spectral_unit_docs

class LazyMask(MaskBase):

    """
    A boolean mask defined by the evaluation of a function on a fixed dataset.

    This is conceptually identical to a fixed boolean mask as in
    :class:`BooleanArrayMask` but defers the
    evaluation of the mask until it is needed.

    Parameters
    ----------
    function : callable
        The function to apply to ``data``. This method should accept
        a numpy array, which will be a subset of the data array passed
        to __init__. It should return a boolean array, where `True` values
        indicate that which pixels are valid/unaffected by masking.
    data : array-like
        The array to evaluate ``function`` on. This should support Numpy-like
        slicing syntax.
    wcs : `~astropy.wcs.WCS`
        The WCS of the input data, which is used to define the coordinates
        for which the boolean mask is defined.
    """

    def __init__(self, function, cube=None, data=None, wcs=None):
        self._function = function
        if cube is not None and (data is not None or wcs is not None):
            raise ValueError("Pass only cube or (data & wcs)")
        elif cube is not None:
            self._data = cube._data
            self._wcs = cube._wcs
        elif data is not None and wcs is not None:
            self._data = data
            self._wcs = wcs
        else:
            raise ValueError("Either a cube or (data & wcs) is required.")

        self._wcs_whitelist = set()

    @property
    def shape(self):
        return self._data.shape

    def _validate_wcs(self, new_data=None, new_wcs=None, **kwargs):
        """
        Check that the new WCS matches the current one

        Parameters
        ----------
        kwargs : dict
            Passed to `wcs_utils.check_equality`
        """
        if new_data is not None:
            if not is_broadcastable_and_smaller(new_data.shape, self._data.shape):
                raise ValueError("data shape cannot be broadcast to match mask shape")
        if new_wcs is not None:
            if new_wcs not in self._wcs_whitelist:
                if not wcs_utils.check_equality(new_wcs, self._wcs,
                                                warn_missing=True, **kwargs):
                    raise ValueError("WCS does not match mask WCS")
                else:
                    self._wcs_whitelist.add(new_wcs)

    def _include(self, data=None, wcs=None, view=()):
        self._validate_wcs(data, wcs)
        return self._function(self._data[view])

    def __getitem__(self, view):
        return LazyMask(self._function, data=self._data[view],
                        wcs=wcs_utils.slice_wcs(self._wcs, view,
                                                shape=self._data.shape,
                                                drop_degenerate=True))

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        """
        Get a LazyMask copy with a WCS in the modified unit
        """
        newwcs = self._get_new_wcs(unit, velocity_convention, rest_value)

        newmask = LazyMask(self._function, data=self._data, wcs=newwcs)
        return newmask

    with_spectral_unit.__doc__ += with_spectral_unit_docs

class LazyComparisonMask(LazyMask):

    """
    A boolean mask defined by the evaluation of a comparison function between a
    fixed dataset and some other value.

    This is conceptually similar to the :class:`LazyMask` but it will ensure
    that the comparison value can be compared to the data

    Parameters
    ----------
    function : callable
        The function to apply to ``data``. This method should accept
        a numpy array, which will be the data array passed to __init__,  and a
        second argument also passed to __init__. It should return a boolean
        array, where `True` values indicate that which pixels are
        valid/unaffected by masking.
    comparison_value : float or array
        The comparison value for the array
    data : array-like
        The array to evaluate ``function`` on. This should support Numpy-like
        slicing syntax.
    wcs : `~astropy.wcs.WCS`
        The WCS of the input data, which is used to define the coordinates
        for which the boolean mask is defined.
    """

    def __init__(self, function, comparison_value, cube=None, data=None,
                 wcs=None):
        self._function = function
        if cube is not None and (data is not None or wcs is not None):
            raise ValueError("Pass only cube or (data & wcs)")
        elif cube is not None:
            self._data = cube._data
            self._wcs = cube._wcs
        elif data is not None and wcs is not None:
            self._data = data
            self._wcs = wcs
        else:
            raise ValueError("Either a cube or (data & wcs) is required.")

        if (hasattr(comparison_value, 'shape') and not
            is_broadcastable_and_smaller(self._data.shape,
                                         comparison_value.shape)):
            raise ValueError("The data and the comparison value cannot "
                             "be broadcast to match shape")

        self._comparison_value = comparison_value

        self._wcs_whitelist = set()

    def _include(self, data=None, wcs=None, view=()):
        self._validate_wcs(data, wcs)

        if hasattr(self._comparison_value, 'shape') and self._comparison_value.shape:
            cv_view = view_of_subset(self._comparison_value.shape,
                                     self._data.shape, view)

            return self._function(self._data[view],
                                  self._comparison_value[cv_view])

        else:
            return self._function(self._data[view],
                                  self._comparison_value)

    def __getitem__(self, view):
        if hasattr(self._comparison_value, 'shape') and self._comparison_value.shape:
            cv_view = view_of_subset(self._comparison_value.shape,
                                     self._data.shape, view)
            return LazyComparisonMask(self._function, data=self._data[view],
                                      comparison_value=self._comparison_value[cv_view],
                                      wcs=wcs_utils.slice_wcs(self._wcs, view,
                                                              drop_degenerate=True))
        else:
            return LazyComparisonMask(self._function, data=self._data[view],
                                      comparison_value=self._comparison_value,
                                      wcs=wcs_utils.slice_wcs(self._wcs, view,
                                                              drop_degenerate=True))

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        """
        Get a LazyComparisonMask copy with a WCS in the modified unit
        """
        newwcs = self._get_new_wcs(unit, velocity_convention, rest_value)

        newmask = LazyComparisonMask(self._function, data=self._data,
                                     comparison_value=self._comparison_value,
                                     wcs=newwcs)
        return newmask

class FunctionMask(MaskBase):

    """
    A mask defined by a function that is evaluated at run-time using the data
    passed to the mask.

    This function differs from :class:`LazyMask` in the arguments which
    are passed to the function. FunctionMasks receive an array,
    wcs object, and view, whereas LazyMasks receive pre-sliced views
    into an array specified at mask-creation time.

    Parameters
    ----------
    function : callable
        The function to evaluate the mask. The call signature should be
        ``function(data, wcs, slice)`` where ``data`` and ``wcs`` are the
        arguments that get passed to e.g. ``include``, ``exclude``,
        ``_filled``, and ``_flattened``. The function should return
        a boolean array, where `True` values indicate that which pixels
        are valid / unaffected by masking.
    """

    def __init__(self, function):
        self._function = function

    def _validate_wcs(self, new_data=None, new_wcs=None, **kwargs):
        pass

    def _include(self, data=None, wcs=None, view=()):
        result = self._function(data, wcs, view)
        if result.shape != data[view].shape:
            raise ValueError("Function did not return mask with correct shape - expected {0}, got {1}".format(data[view].shape, result.shape))
        return result

    def __getitem__(self, slice):
        return self

    def with_spectral_unit(self, unit, velocity_convention=None, rest_value=None):
        """
        Functional masks do not have WCS defined, so this simply returns a copy
        of the current mask in order to be consistent with
        ``with_spectral_unit`` from other Masks
        """
        return FunctionMask(self._function)
