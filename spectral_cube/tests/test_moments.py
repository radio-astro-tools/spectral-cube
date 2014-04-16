import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from ..io import fits as spfits

def moment_cube():
    cube = np.arange(27).reshape([3,3,3])
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN','DEC--TAN','VELO']
    wcs.wcs.cdelt = np.array([-1,1,1], dtype='float32')
    wcs.wcs.crpix = np.array([1,1,1], dtype='float32')
    wcs.wcs.crval = np.array([5,5,5], dtype='float32')
    wcs.wcs.cunit = ['deg','deg','km/s']
    
    hdu = fits.PrimaryHDU(data=cube, header=wcs.to_header())
    return hdu

def test_moment0():
    mc_hdu = moment_cube()
    sc = spfits.load_fits_hdu(mc_hdu)

    mom0_0_np = mc_hdu.data.mean(axis=0)
    mom0_0_sc = sc.moment(0, axis=0)
    
    assert np.testing.assert_array_almost_equal(mom0_0_np, mom0_0_sc)
