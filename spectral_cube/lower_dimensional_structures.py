from __future__ import print_function, absolute_import, division

import warnings
from astropy import units as u
from astropy import wcs
from astropy.io.fits import Header, Card, HDUList, PrimaryHDU
from .io.core import determine_format
from . import spectral_axis
from .utils import SliceWarning
from .cube_utils import convert_bunit

import numpy as np
from astropy import convolution

from .base_class import (BaseNDClass, SpectralAxisMixinClass,
                         SpatialCoordMixinClass)
from . import cube_utils

__all__ = ['LowerDimensionalObject', 'Projection', 'Slice', 'OneDSpectrum']


class LowerDimensionalObject(u.Quantity, BaseNDClass):
    """
    Generic class for 1D and 2D objects.
    """

    @property
    def header(self):
        header = self._header
        # This inplace update is OK; it's not bad to overwrite WCS in this
        # header
        if self.wcs is not None:
            header.update(self.wcs.to_header())
        header['BUNIT'] = self.unit.to_string(format='fits')
        for keyword in header:
            if 'NAXIS' in keyword:
                del header[keyword]
        header.insert(2, Card(keyword='NAXIS', value=self.ndim))
        for ind,sh in enumerate(self.shape[::-1]):
            header.insert(3+ind, Card(keyword='NAXIS{0:1d}'.format(ind+1),
                                      value=sh))

        if 'beam' in self.meta:
            header.update(self.meta['beam'].to_header_keywords())

        return header

    @property
    def hdu(self):
        if self.wcs is None:
            hdu = PrimaryHDU(self.value)
        else:
            hdu = PrimaryHDU(self.value, header=self.wcs.to_header())
        hdu.header['BUNIT'] = self.unit.to_string(format='fits')

        if 'beam' in self.meta:
            hdu.header.update(self.meta['beam'].to_header_keywords())

        return hdu

    def write(self, filename, format=None, overwrite=False):
        """
        Write the lower dimensional object to a file.

        Parameters
        ----------
        filename : str
            The path to write the file to
        format : str
            The kind of file to write. (Currently limited to 'fits')
        overwrite : bool
            If True, overwrite ``filename`` if it exists
        """
        if format is None:
            format = determine_format(filename)
        if format == 'fits':
            self.hdu.writeto(filename, clobber=overwrite)
        else:
            raise ValueError("Unknown format '{0}' - the only available "
                             "format at this time is 'fits'")

    def to(self, unit, equivalencies=[]):
        """
        Return a new `LowerDimensionalObject` of the same class with the
        specified unit.

        See `astropy.units.Quantity.to` for further details.
        """
        converted_array = u.Quantity.to(self, unit,
                                        equivalencies=equivalencies).value

        # use private versions of variables, not the generated property
        # versions
        # Not entirely sure the use of __class__ here is kosher, but we do want
        # self.__class__, not super()
        new = self.__class__(value=converted_array, unit=unit, copy=True,
                             wcs=self._wcs, meta=self._meta, mask=self._mask,
                             header=self._header)

        return new

    def __getitem__(self, key, **kwargs):
        """
        Return a new ``LowerDimensionalObject'' of the same class while keeping
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
                             mask=self._mask,
                             header=self._header,
                             **kwargs)

        return new

    def __array_finalize__(self, obj):
        self._wcs = getattr(obj, '_wcs', None)
        self._meta = getattr(obj, '_meta', None)
        self._mask = getattr(obj, '_mask', None)
        self._header = getattr(obj, '_header', None)
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
    def quantity(self):
        """
        Get a pure `astropy.units.Quantity` representation of the LDO.
        """
        return u.Quantity(self)

class Projection(LowerDimensionalObject, SpatialCoordMixinClass):

    def __new__(cls, value, unit=None, dtype=None, copy=True, wcs=None,
                meta=None, mask=None, header=None, beam=None,
                read_beam=False):

        if np.asarray(value).ndim != 2:
            raise ValueError("value should be a 2-d array")

        if wcs is not None and wcs.wcs.naxis != 2:
            raise ValueError("wcs should have two dimension")

        self = u.Quantity.__new__(cls, value, unit=unit, dtype=dtype,
                                  copy=copy).view(cls)
        self._wcs = wcs
        self._meta = {} if meta is None else meta
        self._mask = mask
        if header is not None:
            self._header = header
        else:
            self._header = Header()

        if beam is not None:
            self._beam = beam
            self.meta['beam'] = beam
        else:
            if "beam" in self.meta:
                self._beam = self.meta['beam']
            elif read_beam:
                beam = cube_utils.try_load_beam(header)
                if beam is not None:
                    self._beam = beam
                    self.meta['beam'] = beam
                else:
                    warnings.warn("Cannot load beam from header.")

        return self

    @property
    def beam(self):
        return self._beam

    @staticmethod
    def from_hdu(hdu):
        '''
        Return a projection from a FITS HDU.
        '''

        if not len(hdu.shape) == 2:
            raise ValueError("HDU must contain two-dimensional data.")

        meta = {}

        mywcs = wcs.WCS(hdu.header)

        if "BUNIT" in hdu.header:
            unit = convert_bunit(hdu.header["BUNIT"])
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
        self.figure = pyplot.imshow(self.value)
        if filename is not None:
            self.figure.savefig(filename)

    def convolve_to(self, beam, convolve=convolution.convolve_fft):
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

        pixscale = wcs.utils.proj_plane_pixel_area(self.wcs.celestial)**0.5 * u.deg

        convolution_kernel = \
            beam.deconvolve(self.beam).as_kernel(pixscale)

        newdata = convolve(self.value, convolution_kernel)

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

        Parameters
        ----------
        [xy]lo/[xy]hi : int or `Quantity` or `min`/`max`
            The endpoints to extract.  If given as a quantity, will be
            interpreted as World coordinates.  If given as a string or
            int, will be interpreted as pixel coordinates.
        """

        self._raise_wcs_no_celestial()

        limit_dict = {'xlo': 0 if xlo == 'min' else xlo,
                      'ylo': 0 if ylo == 'min' else ylo,
                      'xhi': self.shape[1] if xhi == 'max' else xhi,
                      'yhi': self.shape[0] if yhi == 'max' else yhi}
        dims = {'x': 1,
                'y': 0}

        for val in (xlo, ylo, xhi, yhi):
            if hasattr(val, 'unit') and not val.unit.is_equivalent(u.degree):
                raise u.UnitsError("The X and Y slices must be specified in "
                                   "degree-equivalent units.")

        for lim in limit_dict:
            limval = limit_dict[lim]
            if hasattr(limval, 'unit'):
                dim = dims[lim[0]]
                sl = [slice(0, 1)]
                sl.insert(dim, slice(None))
                spine = self.world[sl][dim]
                val = np.argmin(np.abs(limval - spine))
                if limval > spine.max() or limval < spine.min():
                    pass
                    # log.warn("The limit {0} is out of bounds."
                    #          "  Using min/max instead.".format(lim))
                if lim[1:] == 'hi':
                    # End-inclusive indexing: need to add one for the high
                    # slice
                    limit_dict[lim] = val + 1
                else:
                    limit_dict[lim] = val

        slices = [slice(limit_dict[xx + 'lo'], limit_dict[xx + 'hi'])
                  for xx in 'yx']

        return self[slices]

# A slice is just like a projection in every way
class Slice(Projection):
    pass


class OneDSpectrum(LowerDimensionalObject,SpectralAxisMixinClass):

    def __new__(cls, value, unit=None, dtype=None, copy=True, wcs=None,
                meta=None, mask=None, header=None, spectral_unit=None,
                beams=None):

        if np.asarray(value).ndim != 1:
            raise ValueError("value should be a 1-d array")

        if wcs is not None and wcs.wcs.naxis != 1:
            raise ValueError("wcs should have two dimension")

        self = u.Quantity.__new__(cls, value, unit=unit, dtype=dtype,
                                  copy=copy).view(cls)
        self._wcs = wcs
        self._meta = {} if meta is None else meta
        self._mask = mask
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

        if beams is not None:
            self.beams = beams

        return self


    @property
    def header(self):
        header = self._header
        # This inplace update is OK; it's not bad to overwrite WCS in this
        # header
        if self.wcs is not None:
            header.update(self.wcs.to_header())
        header['BUNIT'] = self.unit.to_string(format='fits')
        header.insert(2, Card(keyword='NAXIS', value=self.ndim))
        for ind,sh in enumerate(self.shape[::-1]):
            header.insert(3+ind, Card(keyword='NAXIS{0:1d}'.format(ind+1),
                                      value=sh))

        # Preserve the spectrum's spectral units
        if self._spectral_unit != u.Unit(header['CUNIT1']):
            spectral_scale = spectral_axis.wcs_unit_scale(self._spectral_unit)
            header['CDELT1'] *= spectral_scale
            header['CRVAL1'] *= spectral_scale
            header['CUNIT1'] = self.unit.to_string(format='FITS')

        return header


    @property
    def spectral_axis(self):
        """
        A `~astropy.units.Quantity` array containing the central values of
        each channel along the spectral axis.
        """

        if self._wcs is None:
            spec_axis = np.arange(self.size) * u.dimensionless_unscaled
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
        ax.plot(self.spectral_axis, self.value, drawstyle=drawstyle, **kwargs)
        ax.set_xlabel(self.wcs.wcs.cunit[0])
        ax.set_ylabel(self.unit)
        if filename is not None:
            pyplot.gcf().savefig(filename)


    def with_spectral_unit(self, unit, velocity_convention=None,
                           rest_value=None):

        newwcs, newmeta = self._new_spectral_wcs(unit,
                                                 velocity_convention=velocity_convention,
                                                 rest_value=rest_value)

        newheader = self._nowcs_header.copy()
        newheader['CUNIT1'] = unit.to_string(format='FITS')


        return OneDSpectrum(value=self.value, unit=self.unit, wcs=newwcs,
                            header=newheader, meta=newmeta, copy=False,
                            spectral_unit=unit)

    def __getitem__(self, key, **kwargs):
        try:
            beams = self.beams[key]
        except (AttributeError,TypeError):
            beams = None

        return super(OneDSpectrum, self).__getitem__(key, beams=beams)

    @property
    def hdu(self):
        if hasattr(self, 'beams'):
            warnings.warn("There are multiple beams for this spectrum that "
                          "are being ignored when creating the HDU.")
        return super(OneDSpectrum, self).hdu

    @property
    def hdulist(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdu = self.hdu

        beamhdu = cube_utils.beams_to_bintable(self.beams)

        return HDUList([hdu, beamhdu])
