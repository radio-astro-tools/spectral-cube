import numpy as np

from astropy.coordinates import StokesCoord, custom_stokes_symbol_mapping, StokesSymbol
from astropy.io.registry import UnifiedReadWriteMethod
from .io.core import StokesSpectralCubeRead, StokesSpectralCubeWrite
from .spectral_cube import SpectralCube, BaseSpectralCube
from . import wcs_utils
from .masks import BooleanArrayMask, is_broadcastable_and_smaller

__all__ = ['StokesSpectralCube']


class StokesSpectralCube(object):
    """
    A class to store a spectral cube with multiple Stokes parameters.

    The individual Stokes cubes can share a common mask in addition to having
    component-specific masks.
    """

    # Custom stokes symbol mapping for non-standard symbols
    _custom_stokes_map = {
        -9: StokesSymbol('RX', 'Custom RX'),
        -10: StokesSymbol('RY', 'Custom RY'),
        -11: StokesSymbol('LX', 'Custom LX'),
        -12: StokesSymbol('LY', 'Custom LY'),
        -13: StokesSymbol('XR', 'Custom XR'),
        -14: StokesSymbol('XL', 'Custom XL'),
        -15: StokesSymbol('YR', 'Custom YR'),
        -16: StokesSymbol('YL', 'Custom YL'),
        -17: StokesSymbol('PP', 'Custom PP'),
        -18: StokesSymbol('PQ', 'Custom PQ'),
        -19: StokesSymbol('QP', 'Custom QP'),
        -20: StokesSymbol('QQ', 'Custom QQ'),
        -21: StokesSymbol('RCircular', 'Custom RCircular'),
        -22: StokesSymbol('LCircular', 'Custom LCircular'),
        -23: StokesSymbol('Linear', 'Custom Linear'),
        -24: StokesSymbol('Ptotal', 'Custom Ptotal'),
        -25: StokesSymbol('Plinear', 'Custom Plinear'),
        -26: StokesSymbol('PFtotal', 'Custom PFtotal'),
        -27: StokesSymbol('PFlinear', 'Custom PFlinear'),
        -28: StokesSymbol('Pangle', 'Custom Pangle'),
    }

    def __init__(self, stokes_data, mask=None, meta=None, fill_value=None):

        self._stokes_data = stokes_data
        self._meta = meta or {}
        self._fill_value = fill_value

        reference = tuple(stokes_data.keys())[0]

        # Validate and map Stokes components using StokesCoord, with custom mapping
        try:
            with custom_stokes_symbol_mapping(self._custom_stokes_map):
                stokes_coord = StokesCoord(list(stokes_data.keys()))
        except Exception as e:
            raise ValueError(f"Invalid Stokes components: {e}")

        for component in stokes_data:
            if not isinstance(stokes_data[component], BaseSpectralCube):
                raise TypeError("stokes_data should be a dictionary of "
                                "SpectralCube objects")

            if not wcs_utils.check_equality(stokes_data[component].wcs,
                                            stokes_data[reference].wcs):
                raise ValueError("All spectral cubes in stokes_data "
                                 "should have the same WCS")

            # Validation is now handled by StokesCoord above

            if stokes_data[component].shape != stokes_data[reference].shape:
                raise ValueError("All spectral cubes should have the same shape")

        self._wcs = stokes_data[reference].wcs
        self._shape = stokes_data[reference].shape
        self._stokes_coord = stokes_coord  # Store StokesCoord instance

        if isinstance(mask, BooleanArrayMask):
            if not is_broadcastable_and_smaller(mask.shape, self._shape):
                raise ValueError("Mask shape is not broadcastable to data shape:"
                                 " {0} vs {1}".format(mask.shape, self._shape))

        # Use StokesCoord to determine stokes_type
        stokes_symbols = set(self._stokes_coord.symbol)
        if stokes_symbols.issubset({'I','Q','U','V'}):
            self._stokes_type = 'SKY_STOKES'
        elif stokes_symbols.issubset({'XX', 'XY', 'YX', 'YY'}):
            self._stokes_type = 'FEED_LINEAR'
        elif stokes_symbols.issubset({'RR', 'RL', 'LR', 'LL'}):
            self._stokes_type = 'FEED_CIRCULAR'
        elif stokes_symbols.issubset({'PP', 'PQ', 'QP', 'QQ'}):
            self._stokes_type = 'FEED_GENERIC'
        else:
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
        # Fix: super().__dir__() returns an iterable, convert to list
        return self.components + list(super(StokesSpectralCube, self).__dir__())

    @property
    def components(self):
        # Return the Stokes symbols using StokesCoord
        return list(self._stokes_coord.symbol)

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

        if inherit_mask:
            if self.mask is not None and mask is not None:
                combined_mask = self.mask & mask
            elif self.mask is not None:
                combined_mask = self.mask
            elif mask is not None:
                combined_mask = mask
            else:
                combined_mask = None
            return self._new_cube_with(mask=combined_mask)
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
        data = None  # Ensure data is always defined
        if self.shape[0] < 4:
            errmsg = "Transformation of a subset of Stokes axes is not yet supported."
            errmsg_template = "Transformation from {} to {} is not yet supported."
            raise NotImplementedError(errmsg)
        elif stokes_basis == "Generic":
            data =  self._stokes_data
        elif self.stokes_type == "Linear":
            if stokes_basis == "Circular":
                errmsg_template = "Transformation from {} to {} is not yet supported."
                errmsg = errmsg_template.format("Linear", "Circular")
                raise NotImplementedError(errmsg)
            elif stokes_basis == "Sky":
                mask = self._mask if isinstance(self._mask, dict) else {}
                data = dict(
                    I = (self._stokes_data['XX'] + self._stokes_data['YY']) / 2,
                    Q = (self._stokes_data['XX'] - self._stokes_data['YY']) / 2,
                    U = (self._stokes_data['XY'] + self._stokes_data['YX']) / 2,
                    V = 1j * (self._stokes_data['XY'] - self._stokes_data['YX']) / 2
                )
            elif stokes_basis == "Linear":
                data = self._stokes_data

        elif self.stokes_type == "Circular":
            if stokes_basis == "Linear":
                errmsg_template = "Transformation from {} to {} is not yet supported."
                errmsg = errmsg_template.format("Circular", "Linear")
                raise NotImplementedError(errmsg)
            elif stokes_basis == "Sky":
                mask = self._mask if isinstance(self._mask, dict) else {}
                data = dict(
                    I = (self._stokes_data['RR'] + self._stokes_data['LL']) / 2,
                    Q = (self._stokes_data['RL'] + self._stokes_data['LR']) / 2,
                    U = 1j * (self._stokes_data['RL'] - self._stokes_data['LR']) / 2,
                    V = (self._stokes_data['RR'] - self._stokes_data['LL']) / 2
                )
            elif stokes_basis == "Circular":
                data = self._stokes_data

        elif self.stokes_type == "Sky":
            if stokes_basis == "Linear":
                mask = self._mask if isinstance(self._mask, dict) else {}
                data = dict(
                    XX = 0.5 * (self._stokes_data['I'] + self._stokes_data['Q']),
                    XY = 0.5 * (self._stokes_data['U'] + 1j * self._stokes_data['V']),
                    YX = 0.5 * (self._stokes_data['U'] - 1j * self._stokes_data['V']),
                    YY = 0.5 * (self._stokes_data['I'] - self._stokes_data['Q'])
                )
            elif stokes_basis == "Circular":
                mask = self._mask if isinstance(self._mask, dict) else {}
                data = dict(
                    RR = 0.5 * (self._stokes_data['I'] + self._stokes_data['V']),
                    RL = 0.5 * (self._stokes_data['Q'] + 1j * self._stokes_data['U']),
                    LR = 0.5 * (self._stokes_data['Q'] - 1j * self._stokes_data['U']),
                    LL = 0.5 * (self._stokes_data['I'] - self._stokes_data['V'])
                )
            elif stokes_basis == "Sky":
                data = self._stokes_data
        if data is None:
            raise NotImplementedError("This stokes basis transformation is not implemented.")
        return self._new_cube_with(stokes_data=data)

    def with_spectral_unit(self, unit, **kwargs):

        stokes_data = {k: self._stokes_data[k].with_spectral_unit(unit, **kwargs)
                       for k in self._stokes_data}

        return self._new_cube_with(stokes_data=stokes_data)

    read = UnifiedReadWriteMethod(StokesSpectralCubeRead)
    write = UnifiedReadWriteMethod(StokesSpectralCubeWrite)
