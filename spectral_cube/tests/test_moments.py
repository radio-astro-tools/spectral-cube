import pytest
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

@pytest.mark.parametrize(('axis',),
                         [(0,),
                          (1,),
                          (2,)])
def test_moment0(axis):
    mc_hdu = moment_cube()
    sc = spfits.load_fits_hdu(mc_hdu)

    mom0_np = mc_hdu.data.mean(axis=axis)
    mom0_sc = sc.moment(0, axis=axis)
    
    np.testing.assert_array_almost_equal_nulp(mom0_np, mom0_sc.value)

@pytest.mark.parametrize(('axis',),
                         [(0,),
                          (1,),
                          (2,)])
def test_moment1(axis):
    mc_hdu = moment_cube()
    sc = spfits.load_fits_hdu(mc_hdu)

    pixcrds = np.array([x.ravel() for x in np.indices(sc.shape)]).T
    world = sc._wcs.wcs_pix2world(pixcrds,0)
    # I don't know why I have to transpose it, but the spectral coordinate
    # NEEDS to go first, so this is how it is...
    w = world[:,2-axis].reshape(sc.shape).T

    mom1_np = (mc_hdu.data*w).sum(axis=axis) / (mc_hdu.data.sum(axis=axis))
    mom1_sc = sc.moment(1, axis=axis)
    
    np.testing.assert_array_almost_equal_nulp(mom1_np, mom1_sc.value)

@pytest.mark.parametrize(('axis',),
                         [(0,),
                          (1,),
                          (2,)])
def test_moment2(axis):
    mc_hdu = moment_cube()
    sc = spfits.load_fits_hdu(mc_hdu)

    pixcrds = np.array([x.ravel() for x in np.indices(sc.shape)]).T
    world = sc._wcs.wcs_pix2world(pixcrds,0)
    # I don't know why I have to transpose it, but the spectral coordinate
    # NEEDS to go first, so this is how it is...
    w = world[:,2-axis].reshape(sc.shape).T

    mom2_np = (mc_hdu.data*(w**2)).sum(axis=axis) / (mc_hdu.data.sum(axis=axis))
    mom2_sc = sc.moment(2, axis=axis)
    
    np.testing.assert_array_almost_equal_nulp(mom2_np, mom2_sc.value)
