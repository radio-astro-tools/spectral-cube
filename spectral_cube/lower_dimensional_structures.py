from astropy import units as u
from astropy import wcs
from astropy.io.fits import PrimaryHDU, ImageHDU, Header, Card, HDUList
from .io.core import determine_format

import numpy as np

class LowerDimensionalObject(u.Quantity):
    """
    Generic class for 1D and 2D objects
    """

    @property
    def wcs(self):
        return self._wcs

    @property
    def meta(self):
        return self._meta

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
        return header

    @property
    def hdu(self):
        from astropy.io import fits
        if self.wcs is None:
            hdu = fits.PrimaryHDU(self.value)
        else:
            hdu = fits.PrimaryHDU(self.value, header=self.wcs.to_header())
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
            If True, overwrite `filename` if it exists
        """
        if format is None:
            format = determine_format(filename)
        if format == 'fits':
            self.hdu.writeto(filename, clobber=overwrite)
        else:
            raise ValueError("Unknown format '{0}' - the only available "
                             "format at this time is 'fits'")

class Projection(LowerDimensionalObject):

    def __new__(cls, value, unit=None, dtype=None, copy=True, wcs=None,
                meta=None, mask=None, header=None):

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

        return self


    def quicklook(self, filename=None, use_aplpy=True):
        """
        Use aplpy to make a quick-look image of the projection.  This will make
        the `FITSFigure` attribute available.

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
                    self.FITSFigure = aplpy.FITSFigure(self.hdu)

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

# A slice is just like a projection in every way
class Slice(Projection):
    pass


class OneDSpectrum(LowerDimensionalObject):

    def __new__(cls, value, unit=None, dtype=None, copy=True, wcs=None,
                meta=None, mask=None, header=None):

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

        return self

    @property
    def spectral_axis(self):
        """
        A `~astropy.units.Quantity` array containing the central values of
        each channel along the spectral axis.
        """
        return self.wcs.wcs_pix2world(np.arange(self.size), 0)[0]

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

