from __future__ import print_function, absolute_import, division

import warnings

import numpy as np
from numpy.ma.core import nomask
import dask.array as da

from astropy import convolution
from astropy import units as u
from astropy import wcs
#from astropy import log
from astropy.io.fits import Header, HDUList, PrimaryHDU, BinTableHDU, FITS_rec
from radio_beam import Beam, Beams
from astropy.io.registry import UnifiedReadWriteMethod

from . import spectral_axis
from .io.core import LowerDimensionalObjectWrite
from .utils import SliceWarning, BeamWarning, SmoothingWarning, FITSWarning, BeamUnitsError
from . import cube_utils
from . import wcs_utils
from .masks import BooleanArrayMask, MaskBase

from .base_class import (BaseNDClass, SpectralAxisMixinClass,
                         SpatialCoordMixinClass, MaskableArrayMixinClass,
                         MultiBeamMixinClass, BeamMixinClass,
                         HeaderMixinClass
                        )

__all__ = ['LowerDimensionalObject', 'Projection', 'Slice', 'OneDSpectrum']
class LowerDimensionalObject(u.Quantity, BaseNDClass, HeaderMixinClass):
    """
    Generic class for 1D and 2D objects.
    """

    def _new_view(self, obj=None, unit=None, finalize=True, **kwargs):
        """
        kwargs are passed to _new_view of other object; only known keyword as of June 2023 is ``propagate_info``
        """
        # FORCE finalization to hack around https://github.com/astropy/astropy/issues/14514#issuecomment-1463935711
        try:
            return super(LowerDimensionalObject, self)._new_view(obj=obj, unit=unit, finalize=True, **kwargs)
        except TypeError:
            return super(LowerDimensionalObject, self)._new_view(obj=obj, unit=unit, **kwargs)

    @property
    def hdu(self):
        if self.wcs is None:
            hdu = PrimaryHDU(self.value)
        else:
            hdu = PrimaryHDU(self.value, header=self.header)
        hdu.header['BUNIT'] = self.unit.to_string(format='fits')

        if self.meta is not None and 'beam' in self.meta:
            hdu.header.update(self.meta['beam'].to_header_keywords())

        return hdu

    def read(self, *args, **kwargs):
        raise NotImplementedError()

    write = UnifiedReadWriteMethod(LowerDimensionalObjectWrite)

    def __getslice__(self, start, end, increment=None):
        # I don't know why this is needed, but apparently one of the inherited
        # classes implements getslice, which forces us to overwrite it
        # I can't find any examples where __getslice__ is actually implemented,
        # though, so this seems like a deep and frightening bug.
        #log.debug("Getting a slice from {0} to {1}".format(start,end))
        return self.__getitem__(slice(start, end, increment))

    def __getitem__(self, key, **kwargs):
        """
        Return a new `~spectral_cube.lower_dimensional_structures.LowerDimensionalObject` of the same class while keeping
        other properties fixed.
        """
        new_qty = super(LowerDimensionalObject, self).__getitem__(key)

        if new_qty.ndim < 2:
            # do not return a projection
            return u.Quantity(new_qty)

        if self._wcs is not None:
            if ((isinstance(key, tuple) and
                 any(isinstance(k, slice) for k in key) and
                 len(key) > self.ndim)):
                # Example cases include: indexing tricks like [:,:,None]
                warnings.warn("Slice {0} cannot be used on this {1}-dimensional"
                              " array's WCS.  If this is intentional, you "
                              " should use this {2}'s ``array``  or ``quantity``"
                              " attribute."
                              .format(key, self.ndim, type(self)),
                              SliceWarning
                             )
                return self.quantity[key]
            else:
                newwcs = self._wcs[key]
        else:
            newwcs = None

        new = self.__class__(value=new_qty.value,
                             unit=new_qty.unit,
                             copy=False,
                             wcs=newwcs,
                             meta=self._meta,
                             mask=(self._mask[key] if self._mask is not nomask
                                   else None),
                             header=self._header,
                             **kwargs)

        new._wcs = newwcs
        new._meta = self._meta
        new._mask=(self._mask[key] if self._mask is not nomask else nomask)
        new._header = self._header

        return new

    def __array_finalize__(self, obj):
        self._wcs = getattr(obj, '_wcs', None)
        self._meta = getattr(obj, '_meta', None)
        self._mask = getattr(obj, '_mask', None)
        self._header = getattr(obj, '_header', None)
        self._spectral_unit = getattr(obj, '_spectral_unit', None)
        self._fill_value = getattr(obj, '_fill_value', np.nan)
        self._wcs_tolerance = getattr(obj, '_wcs_tolerance', 0.0)

        if isinstance(obj, VaryingResolutionOneDSpectrum):
            self._beams = getattr(obj, '_beams', None)
        else:
            self._beam = getattr(obj, '_beam', None)

        super(LowerDimensionalObject, self).__array_finalize__(obj)

    @property
    def __array_priority__(self):
        return super(LowerDimensionalObject, self).__array_priority__*2

    @property
    def array(self):
        """
        Get a pure array representation of the LDO.  Useful when multiplying
        and using numpy indexing tricks.
        """
        return np.asarray(self)

    @property
    def _data(self):
        # the _data property is required by several other mixins
        # (which probably means defining it here is a bad design)
        return self.array

    @property
    def quantity(self):
        """
        Get a pure `~astropy.units.Quantity` representation of the LDO.
        """
        return u.Quantity(self)

    def to(self, unit, equivalencies=[], freq=None):
        """
        Return a new `~spectral_cube.lower_dimensional_structures.Projection`
        of the same class with the specified unit.

        See `astropy.units.Quantity.to` for further details.
        """

        if not isinstance(unit, u.Unit):
            unit = u.Unit(unit)

        if unit == self.unit:
            # No copying
            return self

        if hasattr(self, 'with_spectral_unit'):
            freq = self.with_spectral_unit(u.Hz).spectral_axis

        if freq is None and 'RESTFRQ' in self.header:
            freq = self.header['RESTFRQ'] * u.Hz

        # Create the tuple of unit conversions needed.
        factor = cube_utils.bunit_converters(self, unit, equivalencies=equivalencies,
                                             freq=freq)

        converted_array = (self.quantity * factor).value

        # use private versions of variables, not the generated property
        # versions
        # Not entirely sure the use of __class__ here is kosher, but we do want
        # self.__class__, not super()
        new = self.__class__(value=converted_array, unit=unit, copy=True,
                             wcs=self._wcs, meta=self._meta, mask=self._mask,
                             header=self._header)

        return new

    @property
    def _mask(self):
        """ Annoying hack to deal with np.ma.core.is_mask failures (I don't
        like using __ but I think it's necessary here)"""
        if self.__mask is None:
            # need this to be *exactly* the numpy boolean False
            return nomask
        return self.__mask

    @_mask.setter
    def _mask(self, value):
        self.__mask = value

    def shrink_mask(self):
        """
        Copy of the numpy masked_array shrink_mask method.  This is essentially
        a hack needed for matplotlib to show images.
        """
        m = self._mask
        if m.ndim and not m.any():
            self._mask = nomask
        return self

    def _initial_set_mask(self, mask):
        """
        Helper tool to validate mask when originally setting it in __new__

        Note that because this is intended to be used in __new__, order
        matters: ``self`` must have ``_wcs``, for example.
        """
        if mask is None:
            mask = BooleanArrayMask(np.ones_like(self.value, dtype=bool),
                                    self._wcs, shape=self.value.shape)
        elif isinstance(mask, np.ndarray):
            if mask.shape != self.value.shape:
                raise ValueError("Mask shape must match the {0} shape."
                                 .format(self.__class__.__name__)
                                )
            mask = BooleanArrayMask(mask, self._wcs, shape=self.value.shape)
        elif isinstance(mask, MaskBase):
            pass
        else:
            raise TypeError("mask of type {} is not a supported mask "
                            "type.".format(type(mask)))

        # Validate the mask before setting
        mask._validate_wcs(new_data=self.value, new_wcs=self._wcs,
                           wcs_tolerance=self._wcs_tolerance)

        self._mask = mask


class Projection(LowerDimensionalObject, SpatialCoordMixinClass,
                 MaskableArrayMixinClass, BeamMixinClass):

    def __new__(cls, value, unit=None, dtype=None, copy=True, wcs=None,
                meta=None, mask=None, header=None, beam=None,
                fill_value=np.nan, read_beam=False, wcs_tolerance=0.0):

        if np.asarray(value).ndim != 2:
            raise ValueError("value should be a 2-d array")

        if wcs is not None and wcs.wcs.naxis != 2:
            raise ValueError("wcs should have two dimensions")

        self = u.Quantity.__new__(cls, value, unit=unit, dtype=dtype,
                                  copy=copy).view(cls)
        self._wcs = wcs
        self._meta = {} if meta is None else meta
        self._wcs_tolerance = wcs_tolerance

        self._initial_set_mask(mask)

        self._fill_value = fill_value
        if header is not None:
            self._header = header
        else:
            self._header = Header()

        if beam is None:
            if "beam" in self.meta:
                beam = self.meta['beam']
            elif read_beam:
                beam = cube_utils.try_load_beam(header)
                if beam is None:
                    warnings.warn("Cannot load beam from header.",
                                  BeamWarning
                                 )

        if beam is not None:
            self.beam = beam
            self.meta['beam'] = beam
            # TODO: Enable header updating when non-celestial slices are
            # properly handled in the WCS object.
            # self._header.update(beam.to_header_keywords())

        self._cache = {}

        return self

    def with_beam(self, beam, raise_error_jybm=True):
        '''
        Attach a new beam object to the Projection.

        Parameters
        ----------
        beam : `~radio_beam.Beam`
            A new beam object.
        '''

        if not isinstance(beam, Beam):
            raise TypeError("beam must be a radio_beam.Beam object.")

        self.check_jybeam_smoothing(raise_error_jybm=raise_error_jybm)

        meta = self.meta.copy()
        meta['beam'] = beam

        return self._new_projection_with(beam=beam, meta=meta)

    def with_fill_value(self, fill_value):
        """
        Create a new :class:`Projection` or :class:`Slice` with a different
        ``fill_value``.
        """
        return self._new_projection_with(fill_value=fill_value)

    @property
    def _new_thing_with(self):
        return self._new_projection_with

    def _new_projection_with(self, data=None, wcs=None, mask=None, meta=None,
                             fill_value=None, spectral_unit=None, unit=None,
                             header=None, wcs_tolerance=None, beam=None,
                             **kwargs):

        data = self._data if data is None else data
        if unit is None and hasattr(data, 'unit'):
            if data.unit != self.unit:
                raise u.UnitsError("New data unit '{0}' does not"
                                   " match unit '{1}'.  You can"
                                   " override this by specifying the"
                                   " `unit` keyword."
                                   .format(data.unit, self.unit))
            unit = data.unit
        elif unit is None:
            unit = self.unit
        elif unit is not None:
            # convert string units to Units
            if not isinstance(unit, u.Unit):
                unit = u.Unit(unit)

            if hasattr(data, 'unit'):
                if u.Unit(unit) != data.unit:
                    raise u.UnitsError("The specified new cube unit '{0}' "
                                       "does not match the input unit '{1}'."
                                       .format(unit, data.unit))
            else:
                data = u.Quantity(data, unit=unit, copy=False)

        wcs = self._wcs if wcs is None else wcs
        mask = self._mask if mask is None else mask
        if meta is None:
            meta = {}
            meta.update(self._meta)
        if unit is not None:
            meta['BUNIT'] = unit.to_string(format='FITS')

        fill_value = self._fill_value if fill_value is None else fill_value

        if beam is None:
            if hasattr(self, 'beam'):
                beam = self.beam

        newproj = self.__class__(value=data, wcs=wcs, mask=mask, meta=meta,
                                 unit=unit, fill_value=fill_value,
                                 header=header or self._header,
                                 wcs_tolerance=wcs_tolerance or self._wcs_tolerance,
                                 beam=beam,
                                 **kwargs)

        return newproj

    @staticmethod
    def from_hdu(hdu, ext=0):
        '''
        Return a projection from a FITS HDU.

        Parameters
        -----------
        ext : int
            The integer index to load when given an :class:`astropy.io.fits.HDUList`.
            Default is 0 (the first HDU in the list.

        '''
        if isinstance(hdu, HDUList):
            hdul = hdu
            hdu = hdul[ext]

        if not len(hdu.data.shape) == 2:
            raise ValueError("HDU must contain two-dimensional data.")

        meta = {}

        mywcs = wcs.WCS(hdu.header)

        if "BUNIT" in hdu.header:
            unit = cube_utils.convert_bunit(hdu.header["BUNIT"])
            meta["BUNIT"] = hdu.header["BUNIT"]
        else:
            unit = None

        beam = cube_utils.try_load_beam(hdu.header)

        self = Projection(hdu.data, unit=unit, wcs=mywcs, meta=meta,
                          header=hdu.header, beam=beam)

        return self

    def quicklook(self, filename=None, use_aplpy=True, aplpy_kwargs={}):
        """
        Use `APLpy <https://pypi.python.org/pypi/APLpy>`_ to make a quick-look
        image of the projection. This will make the ``FITSFigure`` attribute
        available.

        If there are unmatched celestial axes, this will instead show an image
        without axis labels.

        Parameters
        ----------
        filename : str or Non
            Optional - the filename to save the quicklook to.
        """
        if use_aplpy:
            try:
                if not hasattr(self, 'FITSFigure'):
                    import aplpy
                    self.FITSFigure = aplpy.FITSFigure(self.hdu,
                                                       **aplpy_kwargs)

                self.FITSFigure.show_grayscale()
                self.FITSFigure.add_colorbar()
                if filename is not None:
                    self.FITSFigure.save(filename)
            except (wcs.InconsistentAxisTypesError, ImportError):
                self._quicklook_mpl(filename=filename)
        else:
            self._quicklook_mpl(filename=filename)

    def _quicklook_mpl(self, filename=None):
        from matplotlib import pyplot
        self.figure = pyplot.gcf()
        self.image = pyplot.imshow(self.value)
        if filename is not None:
            self.figure.savefig(filename)

    def convolve_to(self, beam, convolve=convolution.convolve_fft,
                    **kwargs):
        """
        Convolve the image to a specified beam.

        Parameters
        ----------
        beam : `radio_beam.Beam`
            The beam to convolve to
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`

        Returns
        -------
        proj : `Projection`
            A Projection convolved to the given ``beam`` object.
        """

        self._raise_wcs_no_celestial()

        if not hasattr(self, 'beam'):
            raise ValueError("No beam is contained in Projection.meta.")

        # Check if the beams are the same.
        if beam == self.beam:
            warnings.warn("The given beam is identical to the current beam. "
                          "Skipping convolution.")
            return self

        pixscale = wcs.utils.proj_plane_pixel_area(self.wcs.celestial)**0.5 * u.deg

        convolution_kernel = \
            beam.deconvolve(self.beam).as_kernel(pixscale)

        newdata = convolve(self.value, convolution_kernel,
                           normalize_kernel=True,
                           **kwargs)

        self = Projection(newdata, unit=self.unit, wcs=self.wcs,
                          meta=self.meta, header=self.header,
                          beam=beam)

        return self

    def reproject(self, header, order='bilinear'):
        """
        Reproject the image into a new header.

        Parameters
        ----------
        header : `astropy.io.fits.Header`
            A header specifying a cube in valid WCS
        order : int or str, optional
            The order of the interpolation (if ``mode`` is set to
            ``'interpolation'``). This can be either one of the following
            strings:

                * 'nearest-neighbor'
                * 'bilinear'
                * 'biquadratic'
                * 'bicubic'

            or an integer. A value of ``0`` indicates nearest neighbor
            interpolation.
        """

        self._raise_wcs_no_celestial()

        try:
            from reproject.version import version
        except ImportError:
            raise ImportError("Requires the reproject package to be"
                              " installed.")

        # Need version > 0.2 to work with cubes
        from distutils.version import LooseVersion
        if LooseVersion(version) < "0.3":
            raise Warning("Requires version >=0.3 of reproject. The current "
                          "version is: {}".format(version))

        from reproject import reproject_interp

        # TODO: Find the minimal footprint that contains the header and only reproject that
        # (see FITS_tools.regrid_cube for a guide on how to do this)

        newwcs = wcs.WCS(header)
        shape_out = [header['NAXIS{0}'.format(i + 1)] for i in range(header['NAXIS'])][::-1]

        newproj, newproj_valid = reproject_interp((self.value,
                                                   self.header),
                                                  newwcs,
                                                  shape_out=shape_out,
                                                  order=order)

        self = Projection(newproj, unit=self.unit, wcs=newwcs,
                          meta=self.meta, header=header,
                          read_beam=True)

        return self

    def subimage(self, xlo='min', xhi='max', ylo='min', yhi='max'):
        """
        Extract a region spatially.

        When spatial WCS dimensions are given as an `~astropy.units.Quantity`,
        the spatial coordinates of the 'lo' and 'hi' corners are solved together.
        This minimizes WCS variations due to the sky curvature when slicing from
        a large (>1 deg) image.

        Parameters
        ----------
        [xy]lo/[xy]hi : int or `astropy.units.Quantity` or `min`/`max`
            The endpoints to extract.  If given as a quantity, will be
            interpreted as World coordinates.  If given as a string or
            int, will be interpreted as pixel coordinates.
        """

        self._raise_wcs_no_celestial()

        # Solve for the spatial pixel indices together
        limit_dict = wcs_utils.find_spatial_pixel_index(self, xlo, xhi, ylo, yhi)

        slices = [slice(limit_dict[xx + 'lo'], limit_dict[xx + 'hi'])
                  for xx in 'yx']

        return self[tuple(slices)]

    def to(self, unit, equivalencies=[], freq=None):
        """
        Return a new `~spectral_cube.lower_dimensional_structures.Projection`
        of the same class with the specified unit.

        See `astropy.units.Quantity.to` for further details.
        """

        return super(Projection, self).to(unit, equivalencies, freq)

# A slice is just like a projection in every way
class Slice(Projection):
    pass


class BaseOneDSpectrum(LowerDimensionalObject, MaskableArrayMixinClass,
                       SpectralAxisMixinClass):
    """
    Properties shared between OneDSpectrum and VaryingResolutionOneDSpectrum.
    """

    def __new__(cls, value, unit=None, dtype=None, copy=True, wcs=None,
                meta=None, mask=None, header=None, spectral_unit=None,
                fill_value=np.nan, wcs_tolerance=0.0):

        #log.debug("Creating a OneDSpectrum with __new__")

        if np.asarray(value).ndim != 1:
            raise ValueError("value should be a 1-d array")

        if wcs is not None and wcs.wcs.naxis != 1:
            raise ValueError("wcs should have one dimension")

        self = u.Quantity.__new__(cls, value, unit=unit, dtype=dtype,
                                  copy=copy).view(cls)
        self._wcs = wcs
        self._meta = {} if meta is None else meta
        self._wcs_tolerance = wcs_tolerance

        self._initial_set_mask(mask)

        self._fill_value = fill_value
        if header is not None:
            self._header = header
        else:
            self._header = Header()

        self._spectral_unit = spectral_unit

        if spectral_unit is None:
            if 'CUNIT1' in self._header:
                self._spectral_unit = u.Unit(self._header['CUNIT1'])
            elif self._wcs is not None:
                self._spectral_unit = u.Unit(self._wcs.wcs.cunit[0])

        return self

    def __repr__(self):
        prefixstr = '<' + self.__class__.__name__ + ' '
        arrstr = np.array2string(self.filled_data[:].value, separator=',',
                                 prefix=prefixstr)
        return '{0}{1}{2:s}>'.format(prefixstr, arrstr, self._unitstr)

    @staticmethod
    def from_hdu(hdu, ext=0):
        '''
        Return a OneDSpectrum from a FITS HDU or HDU list.

        Parameters
        -----------
        ext : int
            The integer index to load when given an :class:`astropy.io.fits.HDUList`.
            Default is 0 (the first HDU in the list.

        '''

        if isinstance(hdu, HDUList):
            hdul = hdu
            hdu = hdul[ext]
        else:
            hdul = HDUList([hdu])

        if not len(hdu.data.shape) == 1:
            raise ValueError("HDU must contain one-dimensional data.")

        meta = {}

        mywcs = wcs.WCS(hdu.header)

        if "BUNIT" in hdu.header:
            unit = cube_utils.convert_bunit(hdu.header["BUNIT"])
            meta["BUNIT"] = hdu.header["BUNIT"]
        else:
            unit = None

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FITSWarning)
            beam = cube_utils.try_load_beams(hdul)
            try:
                beams = beam
                _ = len(beams)
            except TypeError:
                # beam is scalar and has no len()
                beams = None

        if beams is not None:
            self = VaryingResolutionOneDSpectrum(hdu.data, unit=unit,
                                                 wcs=mywcs, meta=meta,
                                                 header=hdu.header,
                                                 beams=beams)
        else:
            beam = cube_utils.try_load_beam(hdu.header)
            self = OneDSpectrum(hdu.data, unit=unit, wcs=mywcs, meta=meta,
                                header=hdu.header, beam=beam)

        return self

    @property
    def header(self):
        header = super(BaseOneDSpectrum, self).header

        # Preserve the spectrum's spectral units
        if 'CUNIT1' in header and self._spectral_unit != u.Unit(header['CUNIT1']):
            spectral_scale = spectral_axis.wcs_unit_scale(self._spectral_unit)
            header['CDELT1'] *= spectral_scale
            header['CRVAL1'] *= spectral_scale
            header['CUNIT1'] = self.spectral_axis.unit.to_string(format='FITS')

        return header

    @property
    def spectral_axis(self):
        """
        A `~astropy.units.Quantity` array containing the central values of
        each channel along the spectral axis.
        """

        if self._wcs is None:
            spec_axis = np.arange(self.size) * u.one
        else:
            spec_axis = self.wcs.wcs_pix2world(np.arange(self.size), 0)[0] * \
                u.Unit(self.wcs.wcs.cunit[0])
            if self._spectral_unit is not None:
                spec_axis = spec_axis.to(self._spectral_unit)

        return spec_axis

    def quicklook(self, filename=None, drawstyle='steps-mid', **kwargs):
        """
        Plot the spectrum with current spectral units in the currently open
        figure

        kwargs are passed to `matplotlib.pyplot.plot`

        Parameters
        ----------
        filename : str or Non
            Optional - the filename to save the quicklook to.
        """
        from matplotlib import pyplot
        ax = pyplot.gca()
        ax.plot(self.spectral_axis, self.filled_data[:].value,
                drawstyle=drawstyle, **kwargs)
        ax.set_xlabel(self.spectral_axis.unit.to_string(format='latex'))
        ax.set_ylabel(self.unit)
        if filename is not None:
            pyplot.gcf().savefig(filename)

    def with_spectral_unit(self, unit, velocity_convention=None,
                           rest_value=None):

        newwcs, newmeta = self._new_spectral_wcs(unit,
                                                 velocity_convention=velocity_convention,
                                                 rest_value=rest_value)

        newheader = self._nowcs_header.copy()
        newheader.update(newwcs.to_header())
        wcs_cunit = u.Unit(newheader['CUNIT1'])
        newheader['CUNIT1'] = unit.to_string(format='FITS')
        newheader['CDELT1'] *= wcs_cunit.to(unit)

        if self._mask is not None:
            newmask = self._mask.with_spectral_unit(unit,
                                                    velocity_convention=velocity_convention,
                                                    rest_value=rest_value)
            newmask._wcs = newwcs
        else:
            newmask = None

        return self._new_spectrum_with(wcs=newwcs, spectral_unit=unit,
                                       mask=newmask, meta=newmeta,
                                       header=newheader)

    def __getitem__(self, key, **kwargs):
        # Ideally, this could just be in VaryingResolutionOneDSpectrum,
        # but it's about the code is about the same length by just
        # keeping it here.
        try:
            kwargs['beams'] = self.beams[key]
        except (AttributeError, TypeError):
            pass

        new_qty = super(BaseOneDSpectrum, self).__getitem__(key)

        if isinstance(key, slice):

            new = self.__class__(value=new_qty.value,
                                 unit=new_qty.unit,
                                 copy=False,
                                 wcs=wcs_utils.slice_wcs(self._wcs, key,
                                                         shape=self.shape),
                                 meta=self._meta,
                                 mask=(self._mask[key]
                                       if self._mask is not nomask
                                       else nomask),
                                 header=self._header,
                                 wcs_tolerance=self._wcs_tolerance,
                                 fill_value=self.fill_value,
                                 **kwargs)

            return new
        else:
            if self._mask is not nomask:
                # Kind of a hack; this is probably inefficient
                bad = self._mask.exclude()[key]
                if isinstance(bad, da.Array):
                    bad = bad.compute()
                new_qty[bad] = np.nan
            return new_qty

    def __getattribute__(self, attrname):
        # This is a hack to handle dimensionality-reducing functions
        # We want spectrum.max() to return a Quantity, not a spectrum
        # Long-term, we really want `OneDSpectrum` to not inherit from
        # `Quantity`, but for now this approach works.... we just have
        # to add more functions to this list.
        if attrname in ('min', 'max', 'std', 'mean', 'sum', 'cumsum',
                        'nansum', 'ptp', 'var'):
            return getattr(self.quantity, attrname)
        else:
            return super(BaseOneDSpectrum, self).__getattribute__(attrname)

    def spectral_interpolate(self, spectral_grid,
                             suppress_smooth_warning=False,
                             fill_value=None):
        """
        Resample the spectrum onto a specific grid

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

        Returns
        -------
        spectrum : OneDSpectrum
        """

        assert spectral_grid.ndim == 1

        inaxis = self.spectral_axis.to(spectral_grid.unit)

        indiff = np.mean(np.diff(inaxis))
        outdiff = np.mean(np.diff(spectral_grid))

        # account for reversed axes
        if outdiff < 0:
            spectral_grid = spectral_grid[::-1]
            outdiff = np.mean(np.diff(spectral_grid))
            outslice = slice(None, None, -1)
        else:
            outslice = slice(None, None, 1)

        specslice = slice(None) if indiff >= 0 else slice(None, None, -1)
        inaxis = inaxis[specslice]
        indiff = np.mean(np.diff(inaxis))

        # insanity checks
        if indiff < 0 or outdiff < 0:
            raise ValueError("impossible.")

        assert np.all(np.diff(spectral_grid) > 0)
        assert np.all(np.diff(inaxis) > 0)

        np.testing.assert_allclose(np.diff(spectral_grid), outdiff,
                                   err_msg="Output grid must be linear")

        if outdiff > 2 * indiff and not suppress_smooth_warning:
            warnings.warn("Input grid has too small a spacing. The data should "
                          "be smoothed prior to resampling.",
                          SmoothingWarning
                         )

        newspec = np.empty([spectral_grid.size], dtype=self.dtype)
        newmask = np.empty([spectral_grid.size], dtype='bool')

        newspec[outslice] = np.interp(spectral_grid.value, inaxis.value,
                                      self.filled_data[specslice].value,
                                      left=fill_value, right=fill_value)
        mask = self.mask.include()

        if all(mask):
            newmask = np.ones([spectral_grid.size], dtype='bool')
        else:
            interped = np.interp(spectral_grid.value,
                                 inaxis.value, mask[specslice]) > 0
            newmask[outslice] = interped

        newwcs = self.wcs.deepcopy()
        newwcs.wcs.crpix[0] = 1
        newwcs.wcs.crval[0] = spectral_grid[0].value if outslice.step > 0 \
            else spectral_grid[-1].value

        newwcs.wcs.cunit[0] = spectral_grid.unit.to_string(format='FITS')
        newwcs.wcs.cdelt[0] = outdiff.value if outslice.step > 0 \
            else -outdiff.value

        newwcs.wcs.set()

        newheader = self._nowcs_header.copy()
        newheader.update(newwcs.to_header())
        wcs_cunit = u.Unit(newheader['CUNIT1'])
        newheader['CUNIT1'] = spectral_grid.unit.to_string(format='FITS')
        newheader['CDELT1'] *= wcs_cunit.to(spectral_grid.unit)

        newbmask = BooleanArrayMask(newmask, wcs=newwcs)

        return self._new_spectrum_with(data=newspec, wcs=newwcs, mask=newbmask,
                                       header=newheader,
                                       spectral_unit=spectral_grid.unit)

    def spectral_smooth(self, kernel,
                        convolve=convolution.convolve,
                        **kwargs):
        """
        Smooth the spectrum

        Parameters
        ----------
        kernel : `~astropy.convolution.Kernel1D`
            A 1D kernel from astropy
        convolve : function
            The astropy convolution function to use, either
            `astropy.convolution.convolve` or
            `astropy.convolution.convolve_fft`
        kwargs : dict
            Passed to the convolve function
        """

        newspec = convolve(self.value, kernel, normalize_kernel=True, **kwargs)

        return self._new_spectrum_with(data=newspec)

    def to(self, unit, equivalencies=[]):
        """
        Return a new `~spectral_cube.lower_dimensional_structures.OneDSpectrum`
        of the same class with the specified unit.
        See `astropy.units.Quantity.to` for further details.
        """

        return super(BaseOneDSpectrum, self).to(unit, equivalencies, freq=None)

    def with_fill_value(self, fill_value):
        """
        Create a new :class:`OneDSpectrum` with a different ``fill_value``.
        """
        return self._new_spectrum_with(fill_value=fill_value)

    @property
    def _new_thing_with(self):
        return self._new_spectrum_with

    def _new_spectrum_with(self, data=None, wcs=None, mask=None, meta=None,
                           fill_value=None, spectral_unit=None, unit=None,
                           header=None, wcs_tolerance=None,
                           **kwargs):

        data = self._data if data is None else data
        if unit is None and hasattr(data, 'unit'):
            if data.unit != self.unit:
                raise u.UnitsError("New data unit '{0}' does not"
                                   " match unit '{1}'.  You can"
                                   " override this by specifying the"
                                   " `unit` keyword."
                                   .format(data.unit, self.unit))
            unit = data.unit
        elif unit is None:
            unit = self.unit
        elif unit is not None:
            # convert string units to Units
            if not isinstance(unit, u.Unit):
                unit = u.Unit(unit)

            if hasattr(data, 'unit'):
                if u.Unit(unit) != data.unit:
                    raise u.UnitsError("The specified new cube unit '{0}' "
                                       "does not match the input unit '{1}'."
                                       .format(unit, data.unit))
            else:
                data = u.Quantity(data, unit=unit, copy=False)

        wcs = self._wcs if wcs is None else wcs
        mask = self._mask if mask is None else mask
        if meta is None:
            meta = {}
            meta.update(self._meta)
        if unit is not None:
            meta['BUNIT'] = unit.to_string(format='FITS')

        fill_value = self._fill_value if fill_value is None else fill_value
        spectral_unit = self._spectral_unit if spectral_unit is None else u.Unit(spectral_unit)

        spectrum = self.__class__(value=data, wcs=wcs, mask=mask, meta=meta,
                                  unit=unit, fill_value=fill_value,
                                  header=header or self._header,
                                  wcs_tolerance=wcs_tolerance or self._wcs_tolerance,
                                  **kwargs)

        spectrum._spectral_unit = spectral_unit

        return spectrum


class OneDSpectrum(BaseOneDSpectrum, BeamMixinClass):

    def __new__(cls, value, beam=None, read_beam=False, **kwargs):
        self = super(OneDSpectrum, cls).__new__(cls, value, **kwargs)

        if beam is None:
            if "beam" in self.meta:
                beam = self.meta['beam']
            elif read_beam:
                beam = cube_utils.try_load_beam(self.header)
                if beam is None:
                    warnings.warn("Cannot load beam from header.",
                                  BeamWarning
                                  )

        if beam is not None:
            self.beam = beam
            self.meta['beam'] = beam

        self._cache = {}

        return self

    def _new_spectrum_with(self, **kwargs):
        beam = kwargs.pop('beam', None)
        if 'beam' in self._meta and beam is None:
            beam = self.beam
        out = super(OneDSpectrum, self)._new_spectrum_with(beam=beam, **kwargs)
        return out

    def with_beam(self, beam, raise_error_jybm=True):
        '''
        Attach a new beam object to the OneDSpectrum.

        Parameters
        ----------
        beam : `~radio_beam.Beam`
            A new beam object.
        '''

        if not isinstance(beam, Beam):
            raise TypeError("beam must be a radio_beam.Beam object.")

        self.check_jybeam_smoothing(raise_error_jybm=raise_error_jybm)

        meta = self.meta.copy()
        meta['beam'] = beam

        return self._new_spectrum_with(beam=beam, meta=meta)


class VaryingResolutionOneDSpectrum(BaseOneDSpectrum, MultiBeamMixinClass):

    def __new__(cls, value, beams=None, read_beam=False, goodbeams_mask=None, **kwargs):
        self = super(VaryingResolutionOneDSpectrum, cls).__new__(cls, value, **kwargs)
        assert hasattr(self, '_fill_value')

        if beams is None:
            if "beams" in self.meta:
                beams = self.meta['beams']
            elif read_beam:
                beams = cube_utils.try_load_beams(self.header)
                if beams is None:
                    warnings.warn("Cannot load beams table from header.",
                                  BeamWarning
                                  )

        if beams is not None:
            if isinstance(beams, BinTableHDU):
                beam_data_table = beams.data
            elif isinstance(beams, FITS_rec):
                beam_data_table = beams
            else:
                beam_data_table = None

            if beam_data_table is not None:
                beams = Beams(major=u.Quantity(beam_data_table['BMAJ'], u.arcsec),
                              minor=u.Quantity(beam_data_table['BMIN'], u.arcsec),
                              pa=u.Quantity(beam_data_table['BPA'], u.deg),
                              meta=[{key: row[key] for key in beam_data_table.names
                                     if key not in ('BMAJ','BPA', 'BMIN')}
                                    for row in beam_data_table],)
            self.beams = beams
            self.meta['beams'] = beams

        if goodbeams_mask is not None:
            self.goodbeams_mask = goodbeams_mask

        self._cache = {}

        return self

    @property
    def hdu(self):
        warnings.warn("There are multiple beams for this spectrum that "
                      "are being ignored when creating the HDU.",
                      BeamWarning
                     )
        return super(VaryingResolutionOneDSpectrum, self).hdu

    @property
    def hdulist(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdu = self.hdu

        beamhdu = cube_utils.beams_to_bintable(self.beams)

        return HDUList([hdu, beamhdu])

    def _new_spectrum_with(self, **kwargs):
        beams = kwargs.pop('beams', self.beams)
        if beams is None:
            beams = self.beams

        VRODS = VaryingResolutionOneDSpectrum
        out = super(VRODS, self)._new_spectrum_with(beams=beams,
                                                    **kwargs)
        return out

    def __array_finalize__(self, obj):
        super(VaryingResolutionOneDSpectrum, self).__array_finalize__(obj)

        self._beams = getattr(obj, '_beams', None)
        if getattr(obj, 'goodbeams_mask', None) is not None:
            # do NOT use the setter here, because we sometimes need to write
            # intermediate size-mismatch things that later get fixed, e.g., in
            # __getitem__ below
            self._goodbeams_mask = getattr(obj, 'goodbeams_mask', None)

    def __getitem__(self, key):
        new_qty = super(VaryingResolutionOneDSpectrum, self).__getitem__(key)

        # use the goodbeams_mask setter here because it checks size
        new_qty.goodbeams_mask = self.goodbeams_mask[key]
        new_qty.beams = self.unmasked_beams[key]

        return new_qty
