"""
Creates several 4D fits images with a single stokes axis,
in various transpositions
"""

from astropy.io import fits
from astropy import wcs
import numpy as np

HEADER_FILENAME = 'header_jybeam.hdr'

def transpose(d, h, axes):
    d = d.transpose(np.argsort(axes))
    h2 = h.copy()

    for i in range(len(axes)):
        for key in ['NAXIS', 'CDELT', 'CRPIX', 'CRVAL', 'CTYPE', 'CUNIT']:
            h2['%s%i' % (key, i + 1)] = h['%s%i' % (key, axes[i] + 1)]

    return d, h2


if __name__ == "__main__":
    np.random.seed(42)

    beams = np.recarray(4, dtype=[('BMAJ', '>f4'), ('BMIN', '>f4'),
                                  ('BPA', '>f4'), ('CHAN', '>i4'),
                                  ('POL', '>i4')])
    beams['BMAJ'] = [0.4,0.3,0.3,0.4] # arcseconds
    beams['BMIN'] = [0.1,0.2,0.2,0.1]
    beams['BPA'] = [0,45,60,30] # degrees
    beams['CHAN'] = [0,1,2,3]
    beams['POL'] = [0,0,0,0]
    beams = fits.BinTableHDU(beams)

    # Single Stokes
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    h['NAXIS1'] = 2
    h['NAXIS2'] = 3
    h['NAXIS3'] = 4
    h['NAXIS4'] = 1
    d = np.random.random((1, 2, 3, 4))

    fits.writeto('advs.fits', d, h, overwrite=True)

    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto('dvsa.fits', d, h, overwrite=True)

    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto('vsad.fits', d, h, overwrite=True)

    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto('sadv.fits', d, h, overwrite=True)

    d, h = transpose(d, h, [0, 2, 1, 3])
    fits.writeto('sdav.fits', d, h, overwrite=True)

    del h['BMAJ'], h['BMIN'], h['BPA']
    # want 4 spectral channels
    d = np.random.random((4, 3, 2, 1))
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto('sdav_beams.fits', overwrite=True)


    # 3D files
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    h['NAXIS1'] = 2
    h['NAXIS2'] = 3
    h['NAXIS3'] = 4
    h['NAXIS'] = 3
    for k in list(h.keys()):
        if k.endswith('4'):
            del h[k]

    d = np.random.random((4, 3, 2))
    fits.writeto('adv.fits', d, h, overwrite=True)

    h['BUNIT'] = 'JY/BEAM'
    fits.writeto('adv_JYBEAM_upper.fits', d, h, overwrite=True)
    h['BUNIT'] = 'Jy/beam'
    fits.writeto('adv_Jybeam_lower.fits', d, h, overwrite=True)
    h['BUNIT'] = ' Jy / beam '
    fits.writeto('adv_Jybeam_whitespace.fits', d, h, overwrite=True)

    bmaj, bmin, bpa = h['BMAJ'], h['BMIN'], h['BPA']
    del h['BMAJ'], h['BMIN'], h['BPA']
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto('adv_beams.fits', overwrite=True)


    h['BUNIT'] = 'K'
    h['BMAJ'] = bmaj
    h['BMIN'] = bmin
    h['BPA'] = bpa
    d, h = transpose(d, h, [2, 0, 1])
    fits.writeto('vad.fits', d, h, overwrite=True)

    d, h = transpose(d, h, [2, 1, 0])
    fits.writeto('vda.fits', d, h, overwrite=True)

    h['BUNIT'] = 'JY/BEAM'
    fits.writeto('vda_JYBEAM_upper.fits', d, h, overwrite=True)
    h['BUNIT'] = 'Jy/beam'
    fits.writeto('vda_Jybeam_lower.fits', d, h, overwrite=True)
    h['BUNIT'] = ' Jy / beam '
    fits.writeto('vda_Jybeam_whitespace.fits', d, h, overwrite=True)

    del h['BMAJ'], h['BMIN'], h['BPA']
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto('vda_beams.fits', overwrite=True)

    # make a version with spatial pixels
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    for k in list(h.keys()):
        if k.endswith('4'):
            del h[k]
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    d = np.arange(2*5*5).reshape((2,5,5))
    fits.writeto('255.fits', d, h, overwrite=True)

    # test cube for convolution, regridding
    d = np.zeros([2,5,5], dtype='float')
    d[0,2,2] = 1.0
    fits.writeto('255_delta.fits', d, h, overwrite=True)

    d = np.zeros([4,5,5], dtype='float')
    d[:,2,2] = 1.0
    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto('455_delta_beams.fits', overwrite=True)

    d = np.zeros([5,2,2], dtype='float')
    d[2,:,:] = 1.0
    fits.writeto('522_delta.fits', d, h, overwrite=True)

    beams = np.recarray(5, dtype=[('BMAJ', '>f4'), ('BMIN', '>f4'),
                                  ('BPA', '>f4'), ('CHAN', '>i4'),
                                  ('POL', '>i4')])
    beams['BMAJ'] = [0.5,0.4,0.3,0.4,0.5] # arcseconds
    beams['BMIN'] = [0.1,0.2,0.3,0.2,0.1]
    beams['BPA'] = [0,45,60,30,0] # degrees
    beams['CHAN'] = [0,1,2,3,4]
    beams['POL'] = [0,0,0,0,0]
    beams = fits.BinTableHDU(beams)

    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto('522_delta_beams.fits', overwrite=True)

    # Make a 2D spatial version
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    for k in list(h.keys()):
        if k.endswith('4') or k.endswith('3'):
            del h[k]
    h['BUNIT'] = 'K'
    d = np.arange(5 * 5).reshape((5, 5))
    fits.writeto('55.fits', d, h, overwrite=True)

    # test cube for convolution, regridding
    d = np.zeros([5, 5], dtype='float')
    d[2, 2] = 1.0
    fits.writeto('55_delta.fits', d, h, overwrite=True)

    # oneD spectra
    d = np.arange(5, dtype='float')
    h = wcs.WCS(fits.Header.fromtextfile(HEADER_FILENAME)).sub([wcs.WCSSUB_SPECTRAL]).to_header()
    fits.writeto('5_spectral.fits', d, h, overwrite=True)

    hdul = fits.HDUList([fits.PrimaryHDU(data=d, header=h),
                         beams])
    hdul.writeto('5_spectral_beams.fits', overwrite=True)
