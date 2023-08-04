# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
from __future__ import print_function, absolute_import, division

import os
from distutils.version import LooseVersion
from astropy.units.equivalencies import pixel_scale

# Import casatools and casatasks here if available as they can otherwise
# cause a segfault if imported later on during tests.
try:
    import casatools
    import casatasks
except ImportError:
    pass

import pytest
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy import units

from astropy.version import version as astropy_version

if astropy_version < '3.0':
    from astropy.tests.pytest_plugins import *
    del pytest_report_header
else:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS


@pytest.fixture(params=[False, True])
def use_dask(request):
    # Fixture to run tests that use this fixture with both SpectralCube and
    # DaskSpectralCube
    return request.param


def pytest_configure(config):

    config.option.astropy_header = True

    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['regions'] = 'regions'
    PYTEST_HEADER_MODULES['APLpy'] = 'aplpy'

HEADER_FILENAME = os.path.join(os.path.dirname(__file__), 'tests',
                               'data', 'header_jybeam.hdr')


def transpose(d, h, axes):
    d = d.transpose(np.argsort(axes))
    h2 = h.copy()

    for i in range(len(axes)):
        for key in ['NAXIS', 'CDELT', 'CRPIX', 'CRVAL', 'CTYPE', 'CUNIT']:
            h2['%s%i' % (key, i + 1)] = h['%s%i' % (key, axes[i] + 1)]

    return d, h2


def prepare_4_beams():
    beams = np.recarray(4, dtype=[('BMAJ', '>f4'), ('BMIN', '>f4'),
                                  ('BPA', '>f4'), ('CHAN', '>i4'),
                                  ('POL', '>i4')])
    beams['BMAJ'] = [0.4,0.3,0.3,0.4] # arcseconds
    beams['BMIN'] = [0.1,0.2,0.2,0.1]
    beams['BPA'] = [0,45,60,30] # degrees
    beams['CHAN'] = [0,1,2,3]
    beams['POL'] = [0,0,0,0]
    beams = fits.BinTableHDU(beams, name='BEAMS')

    beams.header['TTYPE1'] = 'BMAJ'
    beams.header['TUNIT1'] = 'arcsec'
    beams.header['TTYPE2'] = 'BMIN'
    beams.header['TUNIT2'] = 'arcsec'
    beams.header['TTYPE3'] = 'BPA'
    beams.header['TUNIT3'] = 'deg'

    return beams


def prepare_4_beams_withfullpol():

    nchan = 4
    npol = 4

    beams = np.recarray(nchan*npol, dtype=[('BMAJ', '>f4'), ('BMIN', '>f4'),
                                           ('BPA', '>f4'), ('CHAN', '>i4'),
                                           ('POL', '>i4')])
    beams['BMAJ'] = [0.4,0.3,0.3,0.4] * npol # arcseconds
    beams['BMIN'] = [0.1,0.2,0.2,0.1] * npol
    beams['BPA'] = [0,45,60,30] * npol # degrees
    beams['CHAN'] = [0,1,2,3] * npol

    pol_codes = []
    for i in range(npol):
        pol_codes.extend([i] * nchan)

    beams['POL'] = pol_codes

    beams = fits.BinTableHDU(beams, name='BEAMS')

    beams.header['TTYPE1'] = 'BMAJ'
    beams.header['TUNIT1'] = 'arcsec'
    beams.header['TTYPE2'] = 'BMIN'
    beams.header['TUNIT2'] = 'arcsec'
    beams.header['TTYPE3'] = 'BPA'
    beams.header['TUNIT3'] = 'deg'
    beams.header['TTYPE4'] = 'CHAN'
    beams.header['TTYPE5'] = 'POL'

    return beams


def prepare_advs_data():
    # Single Stokes
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    h['NAXIS1'] = 2
    h['NAXIS2'] = 3
    h['NAXIS3'] = 4
    h['NAXIS4'] = 1
    np.random.seed(42)
    d = np.random.random((1, 2, 3, 4))
    return d, h


def prepare_advs_fullstokes_data():
    # Full Stokes
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    h['NAXIS1'] = 2
    h['NAXIS2'] = 3
    h['NAXIS3'] = 4
    h['NAXIS4'] = 4

    # Add the most basic stokes information to the header
    h['CTYPE4'] = 'STOKES'
    h['CRVAL4'] = 1.0
    h['CDELT4'] = 1.0
    h['CRPIX4'] = 1.0
    h['CUNIT4'] = ''

    np.random.seed(42)
    d = np.random.random((4, 4, 3, 2))
    return d, h


@pytest.fixture
def data_advs(tmp_path):
    d, h = prepare_advs_data()
    fits.writeto(tmp_path / 'advs.fits', d, h)
    return tmp_path / 'advs.fits'


@pytest.fixture
def data_dvsa(tmp_path):
    d, h = prepare_advs_data()
    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto(tmp_path / 'dvsa.fits', d, h)
    return tmp_path / 'dvsa.fits'

@pytest.fixture
def data_vsad(tmp_path):
    d, h = prepare_advs_data()
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto(tmp_path / 'vsad.fits', d, h)
    return tmp_path / 'vsad.fits'

@pytest.fixture
def data_sadv(tmp_path):
    d, h = prepare_advs_data()
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto(tmp_path / 'sadv.fits', d, h)
    return tmp_path / 'sadv.fits'


@pytest.fixture
def data_sdav(tmp_path):
    d, h = prepare_advs_data()
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [0, 2, 1, 3])
    fits.writeto(tmp_path / 'sdav.fits', d, h)
    return tmp_path / 'sdav.fits'


@pytest.fixture
def data_sdav_beams_nounits(tmp_path):
    """
    For testing io when units are not specified
    (they should be arcsec by default
    """
    d, h = prepare_advs_data()
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [0, 2, 1, 3])
    del h['BMAJ'], h['BMIN'], h['BPA']
    # want 4 spectral channels
    np.random.seed(42)
    d = np.random.random((4, 3, 2, 1))
    beams = prepare_4_beams()
    for ii in (1,2,3):
        del beams.header[f'TUNIT{ii}']
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / 'sdav_beams_nounits.fits')
    return tmp_path / 'sdav_beams_nounits.fits'


@pytest.fixture
def data_sdav_beams(tmp_path):
    d, h = prepare_advs_data()
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [1, 2, 3, 0])
    d, h = transpose(d, h, [0, 2, 1, 3])
    del h['BMAJ'], h['BMIN'], h['BPA']
    # want 4 spectral channels
    np.random.seed(42)
    d = np.random.random((4, 3, 2, 1))
    beams = prepare_4_beams()
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / 'sdav_beams.fits')
    return tmp_path / 'sdav_beams.fits'


@pytest.fixture
def data_advs_nobeam(tmp_path):
    d, h = prepare_advs_data()
    del h['BMAJ']
    del h['BMIN']
    del h['BPA']
    fits.writeto(tmp_path / 'advs_nobeam.fits', d, h)
    return tmp_path / 'advs_nobeam.fits'


@pytest.fixture
def data_advs_beams_fullstokes(tmp_path):
    d, h = prepare_advs_fullstokes_data()

    beams = prepare_4_beams_withfullpol()

    hdu = fits.HDUList()
    hdu.append(fits.PrimaryHDU(d, h))
    hdu.append(beams)

    hdu.writeto(tmp_path / 'advs_beams_fullstokes.fits')

    return tmp_path / 'advs_beams_fullstokes.fits'


def prepare_adv_data():
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    h['NAXIS1'] = 2
    h['NAXIS2'] = 3
    h['NAXIS3'] = 4
    h['NAXIS'] = 3
    for k in list(h.keys()):
        if k.endswith('4'):
            del h[k]
    np.random.seed(96)
    d = np.random.random((4, 3, 2))
    return d, h


@pytest.fixture
def data_adv(tmp_path):
    d, h = prepare_adv_data()
    fits.writeto(tmp_path / 'adv.fits', d, h)
    return tmp_path / 'adv.fits'


@pytest.fixture
def data_adv_simple(tmp_path):
    d, h = prepare_adv_data()
    d.flat[:] = np.arange(d.size)
    fits.writeto(tmp_path / 'adv_simple.fits', d, h)
    return tmp_path / 'adv_simple.fits'


@pytest.fixture
def data_adv_jybeam_upper(tmp_path):
    d, h = prepare_adv_data()
    h['BUNIT'] = 'JY/BEAM'
    fits.writeto(tmp_path / 'adv_JYBEAM_upper.fits', d, h)
    return tmp_path / 'adv_JYBEAM_upper.fits'


@pytest.fixture
def data_adv_jybeam_lower(tmp_path):
    d, h = prepare_adv_data()
    h['BUNIT'] = 'Jy/beam'
    fits.writeto(tmp_path / 'adv_Jybeam_lower.fits', d, h)
    return tmp_path / 'adv_Jybeam_lower.fits'


@pytest.fixture
def data_adv_jybeam_whitespace(tmp_path):
    d, h = prepare_adv_data()
    h['BUNIT'] = ' Jy / beam '
    fits.writeto(tmp_path / 'adv_Jybeam_whitespace.fits', d, h)
    return tmp_path / 'adv_Jybeam_whitespace.fits'


@pytest.fixture
def data_adv_beams(tmp_path):
    d, h = prepare_adv_data()
    bmaj, bmin, bpa = h['BMAJ'], h['BMIN'], h['BPA']
    del h['BMAJ'], h['BMIN'], h['BPA']
    beams = prepare_4_beams()
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / 'adv_beams.fits')
    return tmp_path / 'adv_beams.fits'


@pytest.fixture
def data_vad(tmp_path):
    d, h = prepare_adv_data()
    d, h = transpose(d, h, [2, 0, 1])
    fits.writeto(tmp_path / 'vad.fits', d, h)
    return tmp_path / 'vad.fits'

@pytest.fixture
def data_vda(tmp_path):
    d, h = prepare_adv_data()
    d, h = transpose(d, h, [2, 0, 1])
    d, h = transpose(d, h, [2, 1, 0])
    fits.writeto(tmp_path / 'vda.fits', d, h)
    return tmp_path / 'vda.fits'


@pytest.fixture
def data_vda_jybeam_upper(tmp_path):
    d, h = prepare_adv_data()
    d, h = transpose(d, h, [2, 0, 1])
    d, h = transpose(d, h, [2, 1, 0])
    h['BUNIT'] = 'JY/BEAM'
    fits.writeto(tmp_path / 'vda_JYBEAM_upper.fits', d, h)
    return tmp_path / 'vda_JYBEAM_upper.fits'


@pytest.fixture
def data_vda_jybeam_lower(tmp_path):
    d, h = prepare_adv_data()
    d, h = transpose(d, h, [2, 0, 1])
    d, h = transpose(d, h, [2, 1, 0])
    h['BUNIT'] = 'Jy/beam'
    fits.writeto(tmp_path / 'vda_Jybeam_lower.fits', d, h)
    return tmp_path / 'vda_Jybeam_lower.fits'


@pytest.fixture
def data_vda_jybeam_whitespace(tmp_path):
    d, h = prepare_adv_data()
    d, h = transpose(d, h, [2, 0, 1])
    d, h = transpose(d, h, [2, 1, 0])
    h['BUNIT'] = ' Jy / beam '
    fits.writeto(tmp_path / 'vda_Jybeam_whitespace.fits', d, h)
    return tmp_path / 'vda_Jybeam_whitespace.fits'


@pytest.fixture
def data_vda_beams(tmp_path):
    d, h = prepare_adv_data()
    d, h = transpose(d, h, [2, 0, 1])
    d, h = transpose(d, h, [2, 1, 0])
    h['BUNIT'] = ' Jy / beam '
    del h['BMAJ'], h['BMIN'], h['BPA']
    beams = prepare_4_beams()
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / 'vda_beams.fits')
    return tmp_path / 'vda_beams.fits'


@pytest.fixture
def data_vda_beams_image(tmp_path):
    d, h = prepare_adv_data()
    d, h = transpose(d, h, [2, 0, 1])
    d, h = transpose(d, h, [2, 1, 0])
    h['BUNIT'] = ' Jy / beam '
    del h['BMAJ'], h['BMIN'], h['BPA']
    beams = prepare_4_beams()
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / 'vda_beams.fits')
    from casatools import image
    ia = image()
    ia.fromfits(infile=tmp_path / 'vda_beams.fits',
                outfile=tmp_path / 'vda_beams.image',
                overwrite=True)
    for (bmaj, bmin, bpa, chan, pol) in beams.data:
        ia.setrestoringbeam(major={'unit': 'arcsec', 'value': bmaj},
                            minor={'unit': 'arcsec', 'value': bmin},
                            pa={'unit': 'deg', 'value': bpa},
                            channel=chan,
                            polarization=pol)
    ia.close()
    return tmp_path / 'vda_beams.image'


def prepare_255_header():
    # make a version with spatial pixels
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    for k in list(h.keys()):
        if k.endswith('4'):
            del h[k]
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    return h


@pytest.fixture
def data_255(tmp_path):
    h = prepare_255_header()
    d = np.arange(2*5*5, dtype='float').reshape((2,5,5))
    fits.writeto(tmp_path / '255.fits', d, h)
    return tmp_path / '255.fits'


@pytest.fixture
def data_255_delta(tmp_path):
    h = prepare_255_header()
    # test cube for convolution, regridding
    d = np.zeros([2,5,5], dtype='float')
    d[0,2,2] = 1.0
    fits.writeto(tmp_path / '255_delta.fits', d, h)
    return tmp_path / '255_delta.fits'


@pytest.fixture
def data_455_delta_beams(tmp_path):
    h = prepare_255_header()
    # test cube for convolution, regridding
    d = np.zeros([4,5,5], dtype='float')
    d[:,2,2] = 1.0
    beams = prepare_4_beams()
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / '455_delta_beams.fits')
    return tmp_path / '455_delta_beams.fits'


@pytest.fixture
def data_455_degree_beams(tmp_path):
    """
    Test cube for AIPS-style beam specfication
    """
    h = prepare_255_header()
    d = np.zeros([4,5,5], dtype='float')
    beams = prepare_4_beams()
    beams.data['BMAJ'] /= 3600
    beams.data['BMIN'] /= 3600
    beams.header['TTYPE1'] = 'BMAJ'
    beams.header['TUNIT1'] = 'DEGREES'
    beams.header['TTYPE2'] = 'BMIN'
    beams.header['TUNIT2'] = 'DEGREES'

    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / '455_degree_beams.fits')
    return tmp_path / '455_degree_beams.fits'


@pytest.fixture
def data_522_delta(tmp_path):
    h = prepare_255_header()
    d = np.zeros([5,2,2], dtype='float')
    d[2,:,:] = 1.0
    fits.writeto(tmp_path / '522_delta.fits', d, h)
    return tmp_path / '522_delta.fits'


def prepare_5_beams():
    beams = np.recarray(5, dtype=[('BMAJ', '>f4'), ('BMIN', '>f4'),
                                ('BPA', '>f4'), ('CHAN', '>i4'),
                                ('POL', '>i4')])
    beams['BMAJ'] = [0.5,0.4,0.3,0.4,0.5] # arcseconds
    beams['BMIN'] = [0.1,0.2,0.3,0.2,0.1]
    beams['BPA'] = [0,45,60,30,0] # degrees
    beams['CHAN'] = [0,1,2,3,4]
    beams['POL'] = [0,0,0,0,0]
    beams = fits.BinTableHDU(beams, name='BEAMS')
    beams.header['TTYPE1'] = 'BMAJ'
    beams.header['TUNIT1'] = 'arcsec'
    beams.header['TTYPE2'] = 'BMIN'
    beams.header['TUNIT2'] = 'arcsec'
    beams.header['TTYPE3'] = 'BPA'
    beams.header['TUNIT3'] = 'deg'
    return beams


@pytest.fixture
def data_522_delta_beams(tmp_path):
    h = prepare_255_header()
    d = np.zeros([5,2,2], dtype='float')
    d[2,:,:] = 1.0
    beams = prepare_5_beams()
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / '522_delta_beams.fits')
    return tmp_path / '522_delta_beams.fits'


def prepare_55_header():
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    for k in list(h.keys()):
        if k.endswith('4') or k.endswith('3'):
            del h[k]
    h['BUNIT'] = 'K'
    return h


@pytest.fixture
def data_55(tmp_path):
    # Make a 2D spatial version
    h = prepare_55_header()
    d = np.arange(5 * 5, dtype='float').reshape((5, 5))
    fits.writeto(tmp_path / '55.fits', d, h)
    return tmp_path / '55.fits'


@pytest.fixture
def data_55_delta(tmp_path):
    # test cube for convolution, regridding
    h = prepare_55_header()
    d = np.zeros([5, 5], dtype='float')
    d[2, 2] = 1.0
    fits.writeto(tmp_path / '55_delta.fits', d, h)
    return tmp_path / '55_delta.fits'


def prepare_5_header():
    h = wcs.WCS(fits.Header.fromtextfile(HEADER_FILENAME)).sub([wcs.WCSSUB_SPECTRAL]).to_header()
    return h


@pytest.fixture
def data_5_spectral(tmp_path):
    # oneD spectra
    h = prepare_5_header()
    d = np.arange(5, dtype='float')
    fits.writeto(tmp_path / '5_spectral.fits', d, h)
    return tmp_path / '5_spectral.fits'

@pytest.fixture
def data_5_spectral_beams(tmp_path):
    h = prepare_5_header()
    d = np.arange(5, dtype='float')
    beams = prepare_5_beams()
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / '5_spectral_beams.fits')
    return tmp_path / '5_spectral_beams.fits'


def prepare_5_beams_with_pixscale(pixel_scale):
    beams = np.recarray(5, dtype=[('BMAJ', '>f4'), ('BMIN', '>f4'),
                                ('BPA', '>f4'), ('CHAN', '>i4'),
                                ('POL', '>i4')])

    pixel_scale = pixel_scale.to(units.arcsec).value

    beams['BMAJ'] = [3.5 * pixel_scale,3 * pixel_scale,3 * pixel_scale,3 * pixel_scale,3 * pixel_scale] # arcseconds
    beams['BMIN'] = [2 * pixel_scale,2.5 * pixel_scale,3 * pixel_scale,2.5 * pixel_scale,2 * pixel_scale]
    beams['BPA'] = [0,45,60,30,0] # degrees
    beams['CHAN'] = [0,1,2,3,4]
    beams['POL'] = [0,0,0,0,0]
    beams = fits.BinTableHDU(beams, name='BEAMS')

    beams.header['TTYPE1'] = 'BMAJ'
    beams.header['TUNIT1'] = 'arcsec'
    beams.header['TTYPE2'] = 'BMIN'
    beams.header['TUNIT2'] = 'arcsec'
    beams.header['TTYPE3'] = 'BPA'
    beams.header['TUNIT3'] = 'deg'

    return beams


@pytest.fixture
def point_source_5_spectral_beams(tmp_path):

    from radio_beam import Beams
    from astropy.convolution import convolve_fft

    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = "Jy/beam"

    d = np.zeros((5, 11, 11), dtype=float)
    d[:, 5, 5] = 1.

    # NOTE: this matches the header. Should take that directly from the header instead of setting.
    pixel_scale = 2. * units.arcsec

    beams = prepare_5_beams_with_pixscale(pixel_scale)

    for i, beam in enumerate(Beams.from_fits_bintable(beams)):
        # Convolve point source to the beams.
        d[i] = convolve_fft(d[i], beam.as_kernel(pixel_scale))

        # Correct for the beam area in Jy/beam
        # So effectively Jy / pixel -> Jy/beam
        pix_to_beam = beam.sr.to(units.arcsec**2) / pixel_scale**2
        d[i] *= pix_to_beam.value

    # Ensure that the scaling is correct. The center pixel should remain ~1.
    np.testing.assert_allclose(d[:, 5, 5], 1., atol=1e-5)

    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto(tmp_path / 'point_source_conv_5_spectral_beams.fits')
    return tmp_path / 'point_source_conv_5_spectral_beams.fits'


@pytest.fixture
def point_source_5_one_beam(tmp_path):

    from radio_beam import Beam
    from astropy.convolution import convolve_fft

    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = "Jy/beam"

    d = np.zeros((5, 11, 11), dtype=float)
    d[:, 5, 5] = 1.

    # NOTE: this matches the header. Should take that directly from the header instead of setting.
    pixel_scale = 2. * units.arcsec

    beam = Beam(3 * pixel_scale)

    beamprops = beam.to_header_keywords()
    for key in beamprops:
        h[key] = beamprops[key]

    for i in range(5):
        # Convolve point source to the beams.
        d[i] = convolve_fft(d[i], beam.as_kernel(pixel_scale))

        # Correct for the beam area in Jy/beam
        # So effectively Jy / pixel -> Jy/beam
        pix_to_beam = beam.sr.to(units.arcsec**2) / pixel_scale**2
        d[i] *= pix_to_beam.value

    # Ensure that the scaling is correct. The center pixel should remain ~1.
    np.testing.assert_allclose(d[:, 5, 5], 1., atol=1e-5)

    hdul = fits.PrimaryHDU(data=d, header=h)
    hdul.writeto(tmp_path / 'point_source_conv_5_one_beam.fits')
    return tmp_path / 'point_source_conv_5_one_beam.fits'
