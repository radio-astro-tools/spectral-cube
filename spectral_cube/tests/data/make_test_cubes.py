"""
Creates several 4D fits images with a single stokes axis,
in various transpositions
"""

from astropy.io import fits
import numpy as np

HEADER_FILENAME = 'header_jybeam.hdr'

def transpose(d, h, axes):
    d = d.transpose(np.argsort(axes))
    h2 = h.copy()

    for i in range(len(axes)):
        for key in ['NAXIS', 'CDELT', 'CRPIX', 'CRVAL', 'CTYPE']:
            h2['%s%i' % (key, i + 1)] = h['%s%i' % (key, axes[i] + 1)]

    return d, h2


if __name__ == "__main__":
    np.random.seed(42)

    # Single Stokes
    h = fits.header.Header.fromtextfile(HEADER_FILENAME)
    h['BUNIT'] = 'K' # Kelvins are a valid unit, JY/BEAM are not: they should be tested separately
    h['NAXIS1'] = 2
    h['NAXIS2'] = 3
    h['NAXIS3'] = 4
    h['NAXIS4'] = 1
    d = np.random.random((1, 2, 3, 4))

    fits.writeto('advs.fits', d, h, clobber=True)

    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto('dvsa.fits', d, h, clobber=True)

    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto('vsad.fits', d, h, clobber=True)

    d, h = transpose(d, h, [1, 2, 3, 0])
    fits.writeto('sadv.fits', d, h, clobber=True)

    d, h = transpose(d, h, [0, 2, 1, 3])
    fits.writeto('sdav.fits', d, h, clobber=True)

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
    fits.writeto('adv.fits', d, h, clobber=True)

    d, h = transpose(d, h, [2, 0, 1])
    fits.writeto('vad.fits', d, h, clobber=True)

    d, h = transpose(d, h, [2, 1, 0])
    fits.writeto('vda.fits', d, h, clobber=True)

    h['BUNIT'] = 'JY/BEAM'
    fits.writeto('vda_JYBEAM_upper.fits', d, h, clobber=True)
    h['BUNIT'] = 'Jy/beam'
    fits.writeto('vda_Jybeam_lower.fits', d, h, clobber=True)
