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

        with custom_stokes_symbol_mapping(self._custom_stokes_map):
            stokes_coord = StokesCoord(list(stokes_data.keys()))
        

        for component in stokes_data:
            if not isinstance(stokes_data[component], BaseSpectralCube):
                raise TypeError("stokes_data should be a dictionary of "
                                "SpectralCube objects")

            if not wcs_utils.check_equality(stokes_data[component].wcs,
                                            stokes_data[reference].wcs):
                raise ValueError("All spectral cubes in stokes_data "
                                 "should have the same WCS")


            if stokes_data[component].shape != stokes_data[reference].shape:
                raise ValueError("All spectral cubes should have the same shape")

        self._wcs = stokes_data[reference].wcs
        self._shape = stokes_data[reference].shape
        self._stokes_coord = stokes_coord  

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

    def __getitem__(self, view):
        if isinstance(view, str):
            if view in self._stokes_data:
                return self._stokes_data[view]
            else:
                raise KeyError("Key {0} does not exist in this cube.".format(view))
        else:
            # Treat as a spatial/spectral slice and apply to all components
            return self._new_cube_with(
                stokes_data={k: self._stokes_data[k][view] for k in self._stokes_data},
            )

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
        return self.components + list(super(StokesSpectralCube, self).__dir__())

    @property
    def components(self):
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
        Transform the Stokes basis of the cube to the one specified if possible.
        Operates on the underlying data arrays, not SpectralCube objects.
        This makes the operation very expensive, so it should be used with care.
        """
        if len(self._stokes_data) < 4:
            errmsg = "Transformation of a subset of Stokes axes is not yet supported."
            raise NotImplementedError(errmsg)

        def get_data(key):
            return self._stokes_data[key].unmasked_data[:]

        def make_cube(array, template_key):
            return SpectralCube(array, wcs=self._wcs, mask=self._stokes_data[template_key].mask)

        if self.stokes_type == "FEED_LINEAR" and stokes_basis == "Sky":
            XX = get_data('XX')
            YY = get_data('YY')
            XY = get_data('XY')
            YX = get_data('YX')
            I = (XX + YY) / 2
            Q = (XX - YY) / 2
            U = (XY + YX) / 2
            V = -1j * (XY - YX) / 2
            data = {
                'I': make_cube(I, 'XX'),
                'Q': make_cube(Q, 'XX'),
                'U': make_cube(U, 'XX'),
                'V': make_cube(V, 'XX'),
            }
        elif self.stokes_type == "FEED_CIRCULAR" and stokes_basis == "Sky":
            RR = get_data('RR')
            LL = get_data('LL')
            RL = get_data('RL')
            LR = get_data('LR')
            I = (RR + LL) / 2
            Q = (RL + LR) / 2
            U = -1j * (RL - LR) / 2
            V = (RR - LL) / 2
            data = {
                'I': make_cube(I, 'RR'),
                'Q': make_cube(Q, 'RR'),
                'U': make_cube(U, 'RR'),
                'V': make_cube(V, 'RR'),
            }
        elif self.stokes_type == "SKY_STOKES" and stokes_basis == "Linear":
            I = get_data('I')
            Q = get_data('Q')
            U = get_data('U')
            V = get_data('V')
            XX = (I + Q) / 2
            YY = (I - Q) / 2
            XY = (U + 1j * V) / 2  
            YX = (U - 1j * V) / 2  
            data = {
                'XX': make_cube(XX, 'I'),
                'XY': make_cube(XY, 'I'),
                'YX': make_cube(YX, 'I'),
                'YY': make_cube(YY, 'I'),
            }
        elif self.stokes_type == "SKY_STOKES" and stokes_basis == "Circular":
            I = get_data('I')
            Q = get_data('Q')
            U = get_data('U')
            V = get_data('V')
            RR = (I + V) / 2
            LL = (I - V) / 2
            RL = (Q + 1j * U) / 2  
            LR = (Q - 1j * U) / 2 
            data = {
                'RR': make_cube(RR, 'I'),
                'RL': make_cube(RL, 'I'),
                'LR': make_cube(LR, 'I'),
                'LL': make_cube(LL, 'I'),
            }
        elif stokes_basis == "Sky" and self.stokes_type == "SKY_STOKES":
            data = self._stokes_data
        elif stokes_basis == "Linear" and self.stokes_type == "FEED_LINEAR":
            data = self._stokes_data
        elif stokes_basis == "Circular" and self.stokes_type == "FEED_CIRCULAR":
            data = self._stokes_data
        elif stokes_basis == "Generic":
            data = self._stokes_data
        else:
            errmsg = f"Transformation from {self.stokes_type} to {stokes_basis} is not yet supported."
            raise NotImplementedError(errmsg)

        return StokesSpectralCube(stokes_data=data, mask=self._mask, meta=self._meta, fill_value=self._fill_value)
    def spectral_slab(self, lo, hi):
        """
        Extract a new cube between two spectral coordinates.

        Parameters
        ----------
        lo, hi : :class:`~astropy.units.Quantity`
            The lower and upper spectral coordinate for the slab range.
        """
        return self._new_cube_with(
            stokes_data={k: self._stokes_data[k].spectral_slab(lo, hi)
                         for k in self._stokes_data},
        )

    def subcube(self, xlo='min', xhi='max', ylo='min', yhi='max',
                zlo='min', zhi='max', rest_value=None):
        """
        Extract a sub-cube spatially and spectrally.

        Parameters
        ----------
        xlo, xhi, ylo, yhi, zlo, zhi : int or :class:`~astropy.units.Quantity` or ``min``/``max``
            The endpoints to extract. Quantity values are interpreted as world
            coordinates; plain ints or ``'min'``/``'max'`` strings as pixel
            coordinates.
        rest_value : :class:`~astropy.units.Quantity`, optional
            The rest frequency/wavelength for spectral conversions.
        """
        return self._new_cube_with(
            stokes_data={k: self._stokes_data[k].subcube(
                             xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                             zlo=zlo, zhi=zhi, rest_value=rest_value)
                         for k in self._stokes_data},
        )

    def subcube_slices_from_mask(self, region_mask, spatial_only=False):
        """
        Given a mask, return the slices corresponding to the minimum subcube
        that encloses the mask.

        Parameters
        ----------
        region_mask : :class:`~spectral_cube.masks.MaskBase` or boolean `numpy.ndarray`
            The mask with appropriate WCS or an ndarray with matched coordinates.
        spatial_only : bool
            Return only slices that affect the spatial dimensions; the spectral
            dimension will be left unchanged.
        """
        reference = self._stokes_data[self.components[0]]
        return reference.subcube_slices_from_mask(region_mask, spatial_only=spatial_only)

    def subcube_from_mask(self, region_mask):
        """
        Given a mask, return the minimal subcube that encloses the mask.

        Parameters
        ----------
        region_mask : :class:`~spectral_cube.masks.MaskBase` or boolean `numpy.ndarray`
            The mask with appropriate WCS or an ndarray with matched coordinates.
        """
        return self._new_cube_with(
            stokes_data={k: self._stokes_data[k].subcube_from_mask(region_mask)
                         for k in self._stokes_data},
        )

    def minimal_subcube(self, spatial_only=False):
        """
        Return the minimum enclosing subcube where the mask is valid.

        Parameters
        ----------
        spatial_only : bool
            Only compute the minimal subcube in the spatial dimensions.
        """
        return self._new_cube_with(
            stokes_data={k: self._stokes_data[k].minimal_subcube(
                             spatial_only=spatial_only)
                         for k in self._stokes_data},
        )

    def subcube_from_regions(self, region_list, allow_empty=False, minimize=True):
        """
        Extract a masked subcube from a list of ``regions.Region`` objects
        (only functions on celestial dimensions).

        Parameters
        ----------
        region_list : list of ``regions.Region``
            The region(s) to extract.
        allow_empty : bool, optional
            If False, raise an exception when the region has no overlap with
            the cube. Default is False.
        minimize : bool, optional
            Run :meth:`~SpectralCube.minimal_subcube` after masking. Default
            is True.
        """
        return self._new_cube_with(
            stokes_data={k: self._stokes_data[k].subcube_from_regions(
                             region_list, allow_empty=allow_empty,
                             minimize=minimize)
                         for k in self._stokes_data},
        )

    def subcube_from_ds9region(self, ds9_region, allow_empty=False):
        """
        Extract a masked subcube from a DS9 region string
        (only functions on celestial dimensions).

        Parameters
        ----------
        ds9_region : str
            The DS9 region(s) to extract.
        allow_empty : bool, optional
            If False, raise an exception when the region has no overlap with
            the cube. Default is False.
        """
        return self._new_cube_with(
            stokes_data={k: self._stokes_data[k].subcube_from_ds9region(
                             ds9_region, allow_empty=allow_empty)
                         for k in self._stokes_data},
        )

    def subcube_from_crtfregion(self, crtf_region, allow_empty=False):
        """
        Extract a masked subcube from a CRTF region string
        (only functions on celestial dimensions).

        Parameters
        ----------
        crtf_region : str
            The CRTF region(s) string to extract.
        allow_empty : bool, optional
            If False, raise an exception when the region has no overlap with
            the cube. Default is False.
        """
        return self._new_cube_with(
            stokes_data={k: self._stokes_data[k].subcube_from_crtfregion(
                             crtf_region, allow_empty=allow_empty)
                         for k in self._stokes_data},
        )

    def with_spectral_unit(self, unit, **kwargs):

        stokes_data = {k: self._stokes_data[k].with_spectral_unit(unit, **kwargs)
                       for k in self._stokes_data}

        return self._new_cube_with(stokes_data=stokes_data)

    read = UnifiedReadWriteMethod(StokesSpectralCubeRead)
    write = UnifiedReadWriteMethod(StokesSpectralCubeWrite)
