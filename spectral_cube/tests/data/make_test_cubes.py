"""
Creates several 4D fits images with a single stokes axis,
in various transpositions
"""

from astropy.io import fits
import numpy as np

HEADER_STR = "SIMPLE  =                    T / Written by IDL:  Fri Feb 20 13:46:36 2009      BITPIX  =                  -32  /                                               NAXIS   =                    4  /                                               NAXIS1  =                 1884  /                                               NAXIS2  =                 2606  /                                               NAXIS3  =                  200 //                                               NAXIS4  =                    1  /                                               EXTEND  =                    T  /                                               BSCALE  =    1.00000000000E+00  /                                               BZERO   =    0.00000000000E+00  /                                               BLANK   =                   -1  /                                               TELESCOP= 'VLA     '  /                                                         CDELT1  =   -5.55555561268E-04  /                                               CRPIX1  =    1.37300000000E+03  /                                               CRVAL1  =    2.31837500515E+01  /                                               CTYPE1  = 'RA---SIN'  /                                                         CDELT2  =    5.55555561268E-04  /                                               CRPIX2  =    1.15200000000E+03  /                                               CRVAL2  =    3.05765277962E+01  /                                               CTYPE2  = 'DEC--SIN'  /                                                         CDELT3  =    1.28821496879E+03  /                                               CRPIX3  =    1.00000000000E+00  /                                               CRVAL3  =   -3.21214698632E+05  /                                               CTYPE3  = 'VELO-HEL'  /                                                         CDELT4  =    1.00000000000E+00  /                                               CRPIX4  =    1.00000000000E+00  /                                               CRVAL4  =    1.00000000000E+00  /                                               CTYPE4  = 'STOKES  '  /                                                         DATE-OBS= '1998-06-18T16:30:25.4'  /                                            RESTFREQ=    1.42040571841E+09  /                                               CELLSCAL= 'CONSTANT'  /                                                         BUNIT   = 'JY/BEAM '  /                                                         EPOCH   =    2.00000000000E+03  /                                               OBJECT  = 'M33     '           /                                                OBSERVER= 'AT206   '  /                                                         VOBS    =   -2.57256763070E+01  /                                               LTYPE   = 'channel '  /                                                         LSTART  =    2.15000000000E+02  /                                               LWIDTH  =    1.00000000000E+00  /                                               LSTEP   =    1.00000000000E+00  /                                               BTYPE   = 'intensity'  /                                                        DATAMIN =   -6.57081836835E-03  /                                               DATAMAX =    1.52362231165E-02  /     "


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
    h = fits.header.Header.fromstring(HEADER_STR)
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
    h = fits.header.Header.fromstring(HEADER_STR)
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
