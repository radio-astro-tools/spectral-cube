from __future__ import print_function, absolute_import, division

import six
import numpy as np

from astropy.io.registry import UnifiedReadWriteMethod
from .io.core import StokesSpectralCubeRead, StokesSpectralCubeWrite
from .spectral_cube import SpectralCube, BaseSpectralCube
from . import wcs_utils
from .masks import BooleanArrayMask, is_broadcastable_and_smaller

__all__ = ['StokesSpectralCube']

VALID_STOKES = ['I', 'Q', 'U', 'V', 'RR', 'LL', 'RL', 'LR', 'XX', 'XY', 'YX', 'YY', 
                'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR', 'YL', 'PP', 'PQ', 'QP', 'QQ', 
                'RCircular', 'LCircular', 'Linear', 'Ptotal', 'Plinear', 'PFtotal', 
                'PFlinear', 'Pangle']
STOKES_SKY = ['I','Q','U','V']
STOKES_FEED_LINEAR = ['XX', 'XY', 'YX', 'YY']
STOKES_FEED_CIRCULAR = ['RR', 'RL', 'LR', 'LL']
STOKES_FEED_GENERIC = ['PP', 'PQ', 'QP', 'QQ']


class StokesSpectralCube(object):
    """
    A class to store a spectral cube with multiple Stokes parameters.

    The individual Stokes cubes can share a common mask in addition to having
    component-specific masks.
    """

    def __init__(self, stokes_data, mask=None, meta=None, fill_value=None):

        self._stokes_data = stokes_data
        self._meta = meta or {}
        self._fill_value = fill_value

        reference = tuple(stokes_data.keys())[0]

        for component in stokes_data:

            if not isinstance(stokes_data[component], BaseSpectralCube):
                raise TypeError("stokes_data should be a dictionary of "
                                "SpectralCube objects")

            if not wcs_utils.check_equality(stokes_data[component].wcs,
                                            stokes_data[reference].wcs):
                raise ValueError("All spectral cubes in stokes_data "
                                 "should have the same WCS")

            if component not in VALID_STOKES:
                raise ValueError("Invalid Stokes component: {0} - should be one of I, Q, U, V, RR, LL, RL, LR, XX, XY, YX, YY, \
                                 RX, RY, LX, LY, XR, XL, YR, YL, PP, PQ, QP, QQ, \
                                 RCircular, LCircular, Linear, Ptotal, Plinear, PFtotal, PFlinear, Pangle".format(component))

            if stokes_data[component].shape != stokes_data[reference].shape:
                raise ValueError("All spectral cubes should have the same shape")

        self._wcs = stokes_data[reference].wcs
        self._shape = stokes_data[reference].shape

        if isinstance(mask, BooleanArrayMask):
            if not is_broadcastable_and_smaller(mask.shape, self._shape):
                raise ValueError("Mask shape is not broadcastable to data shape:"
                                 " {0} vs {1}".format(mask.shape, self._shape))

        if set(stokes_data).issubset(set(STOKES_SKY)):
            self._stokes_type = 'SKY_STOKES'
        elif set(stokes_data).issubset(set(STOKES_FEED_LINEAR)):
            self._stokes_type = 'FEED_LINEAR'
        elif set(stokes_data).issubset(set(STOKES_FEED_CIRCULAR)):
            self._stokes_type = 'FEED_CIRCULAR'
        elif set(stokes_data).issubset(set(STOKES_FEED_GENERIC)):
            self._stokes_type = 'FEED_GENERIC'
        elif set(stokes_data).issubset(set(VALID_STOKES)):
            self._stokes_type = 'VALID_STOKES'

        self._mask = mask

    def __getitem__(self, key):
        if key in self._stokes_data:
            return self._stokes_data[key]
        else:
            raise KeyError("Key {0} does not exist in this cube.".format(key))

    def __setitem__(self, key, item):
        if key in self._stokes_data:
            self._stokes_data[key] = item
        else:
            errmsg = "Assigning new Stokes axes is not yet supported."
            raise NotImplementedError(errmsg)

    @property
    def shape(self):
        return self._shape

    @property
    def stokes_data(self):
        """
        The underlying data
        """
        return self._stokes_data

    @property
    def mask(self):
        """
        The underlying mask
        """
        return self._mask

    @property
    def wcs(self):
        return self._wcs

    def __dir__(self):
        if six.PY2:
            return self.components + dir(type(self)) + list(self.__dict__)
        else:
            return self.components + super(StokesSpectralCube, self).__dir__()

    @property
    def components(self):
        return list(self._stokes_data.keys())

    @property
    def stokes_type(self):
        """
        Defines the type of stokes that has been setup.  `stokes_type` can be sky, linear, circular, generic, or other.
          * `Sky` refers to stokes in sky basis, I,Q,U,V
          * `Linear` refers to the stokes in linear feed basis, XX, XY, YX, YY
          * `Circular` refers to stokes in circular feed basis, RR, RL, LR, LL
          * `Generic` refers to the general four orthogonal components, PP, PQ, QP, QQ
        """
        return self._stokes_type

    def __getattr__(self, attribute):
        """
        Descriptor to return the Stokes cubes
        """
        if attribute in self._stokes_data:
            if self.mask is not None:
                return self._stokes_data[attribute].with_mask(self.mask)
            else:
                return self._stokes_data[attribute]
        else:
            raise AttributeError("StokesSpectralCube has no attribute {0}".format(attribute))

    def with_mask(self, mask, inherit_mask=True):
        """
        Return a new StokesSpectralCube instance that contains a composite mask
        of the current StokesSpectralCube and the new ``mask``.

        Parameters
        ----------
        mask : :class:`MaskBase` instance, or boolean numpy array
            The mask to apply. If a boolean array is supplied,
            it will be converted into a mask, assuming that
            `True` values indicate included elements.

        inherit_mask : bool (optional, default=True)
            If True, combines the provided mask with the
            mask currently attached to the cube

        Returns
        -------
        new_cube : :class:`StokesSpectralCube`
            A cube with the new mask applied.

        Notes
        -----
        This operation returns a view into the data, and not a copy.
        """
        if isinstance(mask, np.ndarray):
            if not is_broadcastable_and_smaller(mask.shape, self.shape):
                raise ValueError("Mask shape is not broadcastable to data shape: "
                                 "%s vs %s" % (mask.shape, self.shape))
            mask = BooleanArrayMask(mask, self.wcs)

        if self._mask is not None:
            return self._new_cube_with(mask=self.mask & mask if inherit_mask else mask)
        else:
            return self._new_cube_with(mask=mask)

    def _new_cube_with(self, stokes_data=None,
                       mask=None, meta=None, fill_value=None):

        data = self._stokes_data if stokes_data is None else stokes_data
        mask = self._mask if mask is None else mask
        if meta is None:
            meta = {}
            meta.update(self._meta)

        fill_value = self._fill_value if fill_value is None else fill_value

        cube = StokesSpectralCube(stokes_data=data, mask=mask,
                                  meta=meta, fill_value=fill_value)

        return cube

    def transform_basis(self, stokes_basis=''):
        """
        The goal of this function is to transform the stokes basis of the cube to
        the one specified if possible. In principle one needs at least two related
        stokes of a given basis for the transformation to be possible. At this moment
        we are limiting it to the cases where all four stokes parameters in a given
        basis be available within the cube. The transformations are as follows
        Linear to Sky
        I = (XX + YY) / 2
        Q = (XX - YY) / 2
        U = (XY + YX) / 2
        V = 1j(XY - YX) / 2

        Circular to Sky
        I = (RR + LL) / 2
        Q = (RL + LR) / 2
        U = 1j*(RL - LR) / 2
        V = (RR - LL) / 2
        """
        if self.shape[0] < 4:
            errmsg = "Transformation of a subset of Stokes axes is not yet supported."
            errmsg_template = "Transformation from {} to {} is not yet supported."

            raise NotImplementedError(errmsg)
        elif stokes_basis == "Generic":
            data =  self._stokes_data
        elif self.stokes_type == "Linear":
            if stokes_basis == "Circular":
                errmsg = errmsg_template.format("Linear", "Circular")
                raise NotImplementedError(errmsg)
            elif stokes_basis == "Sky":
                data = dict(I = self.__class__(self._stokes_data['XX'] + self._stokes_data['YY'], wcs = self.wcs, mask=self._mask['XX']&self._mask['YY']),
                            Q = self.__class__(self._stokes_data['XX'] - self._stokes_data['YY'], wcs = self.wcs, mask=self._mask['XX']&self._mask['YY']),
                            U = self.__class__(self._stokes_data['XY'] + self._stokes_data['YX'], wcs = self.wcs, mask=self._mask['XY']&self._mask['YX']),
                            V = self.__class__(1j*(self._stokes_data['XY'] - self._stokes_data['YX']), wcs = self.wcs, mask=self._mask['XY']&self._mask['YX']))
            elif stokes_basis == "Linear":
                data = self._stokes_data

        elif self.stokes_type == "Circular":
            if stokes_basis == "Linear":
                errmsg = errmsg_template.format("Circular", "Linear")
                raise NotImplementedError(errmsg)
            elif stokes_basis == "Sky":
                data = dict(I = self.__class__(self._stokes_data['RR'] + self._stokes_data['LL'], wcs = self.wcs, mask=self._mask['RR']&self._mask['LL']),
                            Q = self.__class__(self._stokes_data['RL'] + self._stokes_data['RL'], wcs = self.wcs, mask=self._mask['RL']&self._mask['LR']),
                            U = self.__class__(1j*(self._stokes_data['RL'] - self._stokes_data['LR']), wcs = self.wcs, mask=self._mask['RL']&self._mask['LR']),
                            V = self.__class__(self._stokes_data['RR'] - self._stokes_data['LL'], wcs = self.wcs, mask=self._mask['RR']&self._mask['LL']))
            elif stokes_basis == "Circular":
                data = self._stokes_data

        elif self.stokes_type == "Sky":
            if stokes_basis == "Linear":
                data = dict(XX = self.__class__(0.5*(self._stokes_data['I'] + self._stokes_data['Q']), wcs = self.wcs, mask=self._mask['I']&self._mask['Q']),
                            XY = self.__class__(0.5*(self._stokes_data['U'] + 1j*self._stokes_data['V']), wcs = self.wcs, mask=self._mask['U']&self._mask['V']),
                            YX = self.__class__(0.5*(self._stokes_data['U'] - 1j*self._stokes_data['V']), wcs = self.wcs, mask=self._mask['U']&self._mask['V']),
                            YY = self.__class__(0.5*(self._stokes_data['I'] - self._stokes_data['Q']), wcs = self.wcs, mask=self._mask['I']&self._mask['Q']))
            elif stokes_basis == "Circular":
                data = dict(RR = self.__class__(0.5*(self._stokes_data['I'] + self._stokes_data['V']), wcs = self.wcs, mask=self._mask['I']&self._mask['V']),
                            RL = self.__class__(0.5*(self._stokes_data['Q'] + 1j*self._stokes_data['U']), wcs = self.wcs, mask=self._mask['Q']&self._mask['U']),
                            LR = self.__class__(0.5*(self._stokes_data['Q'] - 1j*self._stokes_data['U']), wcs = self.wcs, mask=self._mask['Q']&self._mask['U']),
                            LL = self.__class__(0.5*(self._stokes_data['I'] - self._stokes_data['V']), wcs = self.wcs, mask=self._mask['I']&self._mask['V']))
            elif stokes_basis == "Sky":
                data = self._stokes_data        
        return self._new_cube_with(stokes_data=data)
        

    def with_spectral_unit(self, unit, **kwargs):

        stokes_data = {k: self._stokes_data[k].with_spectral_unit(unit, **kwargs)
                       for k in self._stokes_data}

        return self._new_cube_with(stokes_data=stokes_data)

    read = UnifiedReadWriteMethod(StokesSpectralCubeRead)
    write = UnifiedReadWriteMethod(StokesSpectralCubeWrite)
