import abc

import numpy as np
from . import wcs_utils

__all__ = ['InvertedMask', 'CompositeMask', 'BooleanArrayMask',
           'LazyMask', 'FunctionMask']


class MaskBase(object):

    __metaclass__ = abc.ABCMeta

    def include(self, data=None, wcs=None, view=()):
        """
        Return a boolean array indicating which values should be included.

        If ``view`` is passed, only the sliced mask will be returned, which
        avoids having to load the whole mask in memory. Otherwise, the whole
        mask is returned in-memory.
        """
        self._validate_wcs(data, wcs)
        return self._include(data=data, wcs=wcs, view=view)

    def _validate_wcs(self, data, wcs):
        """
        This method can be overridden in cases where the data and WCS have to
        conform to some rules. This gets called automatically when
        ``include`` or ``exclude`` are called.
        """
        pass

    @abc.abstractmethod
    def _include(self, data=None, wcs=None, view=()):
        pass

    def exclude(self, data=None, wcs=None, view=()):
        """
        Return a boolean array indicating which values should be excluded.

        If ``view`` is passed, only the sliced mask will be returned, which
        avoids having to load the whole mask in memory. Otherwise, the whole
        mask is returned in-memory.
        """
        self._validate_wcs(data, wcs)
        return self._exclude(data=data, wcs=wcs, view=view)

    def _exclude(self, data=None, wcs=None, view=()):
        return ~self._include(data=data, wcs=wcs, view=view)

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
        return data[view][self.include(data=data, wcs=wcs, view=view)]

    def _filled(self, data, wcs=None, fill=np.nan, view=()):
        """
        Replace the exluded elements of *array* with *fill*.

        Parameters
        ----------
        data : array-like
            Input array
        fill : number
            Replacement value
        view : tuple, optional
            Any slicing to apply to the data before flattening

        Returns
        -------
        filled_array : `~numpy.ndarray`
            A 1-D ndarray containing the filled output

        Notes
        -----
        This is an internal method used by :class:`SpectralCube`.
        Users should use the property :meth:`BooleanArrayMask.filled_data`
        """
        sliced_data = data[view].copy().astype(np.float)
        ex = self.exclude(data=data, wcs=wcs, view=view)
        sliced_data[ex] = fill
        return sliced_data

    def __and__(self, other):
        return CompositeMask(self, other, operation='and')

    def __or__(self, other):
        return CompositeMask(self, other, operation='or')

    def __invert__(self):
        return InvertedMask(self)

    def __getitem__(self):
        raise NotImplementedError("Slicing not supported by mask class {0}".format(self.__class__.__name__))


class InvertedMask(MaskBase):

    def __init__(self, mask):
        self._mask = mask

    def _include(self, data=None, wcs=None, view=()):
        return ~self._mask.include(data=data, wcs=wcs, view=view)

    def __getitem__(self, view):
        return InvertedMask(self._mask[view])


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

    def _include(self, data=None, wcs=None, view=()):
        result_mask_1 = self._mask1._include(data=data, wcs=wcs, view=view)
        result_mask_2 = self._mask2._include(data=data, wcs=wcs, view=view)
        if self._operation == 'and':
            return result_mask_1 & result_mask_2
        elif self._operation == 'or':
            return result_mask_1 | result_mask_2
        else:
            raise ValueError("Operation '{0}' not supported".format(self._operation))

    def __getitem__(self, view):
        return CompositeMask(self._mask1[view], self._mask2[view], operation=self._operation)


class BooleanArrayMask(MaskBase):

    """
    A mask defined as an array on a spectral cube WCS
    """

    def __init__(self, mask, wcs, include=True):
        self._mask = mask
        self._mask_type = 'include' if include else 'exclude'
        self._wcs = wcs
        self._wcs_whitelist = set()

    def _validate_wcs(self, new_data, new_wcs):
        if new_data.shape != self._mask.shape:
            raise ValueError("data shape does not match mask shape")
        if new_wcs not in self._wcs_whitelist:
            if str(new_wcs.to_header()) != str(self._wcs.to_header()):
                raise ValueError("WCS does not match mask WCS")
        self._wcs_whitelist.add(new_wcs)

    def _include(self, data=None, wcs=None, view=()):
        result_mask = self._mask[view]
        return result_mask if self._mask_type == 'include' else ~result_mask

    def _exclude(self, data=None, wcs=None, view=()):
        result_mask = self._mask[view]
        return result_mask if self._mask_type == 'exclude' else ~result_mask

    @property
    def shape(self):
        return self._include_mask.shape

    def __getitem__(self, view):
        return BooleanArrayMask(self._mask[view], wcs_utils.slice_wcs(self._wcs, view))


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
        to __init__. It should return a boolean array, where True values
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

    def _validate_wcs(self, new_data, new_wcs):
        if new_data.shape != self._data.shape:
            raise ValueError("data shape does not match mask shape")
        if new_wcs not in self._wcs_whitelist:
            if str(new_wcs.to_header()) != str(self._wcs.to_header()):
                raise ValueError("WCS does not match mask WCS")
        self._wcs_whitelist.add(new_wcs)

    def _include(self, data=None, wcs=None, view=()):
        self._validate_wcs(data, wcs)
        return self._function(self._data[view])

    def __getitem__(self, view):
        return LazyMask(self._function, data=self._data[view], wcs=wcs_utils.slice_wcs(self._wcs, view))


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

    def _validate_wcs(self, data, wcs):
        pass

    def _include(self, data=None, wcs=None, view=()):
        result = self._function(data, wcs, view)
        if result.shape != data[view].shape:
            raise ValueError("Function did not return mask with correct shape - expected {0}, got {1}".format(data[view].shape, result.shape))
        return result

    def __getitem__(self, slice):
        return self
