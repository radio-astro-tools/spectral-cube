from ..spectral_axis import convert_spectral_axis
from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy import constants
from astropy.tests.helper import pytest
import os
import numpy as np

def data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, filename)

def test_cube_wcs_freqtovel():
    header = fits.Header.fromtextfile(data_path('cubewcs1.hdr'))
    w1 = wcs.WCS(header)

    newwcs = convert_spectral_axis(w1, 'km/s', 'VRAD')
    assert newwcs.wcs.ctype[2] == 'VRAD'
    assert newwcs.wcs.crval[2] == 305.2461585938794
    assert newwcs.wcs.cunit[2] == u.Unit('km/s')

def test_cube_wcs_freqtovopt():
    header = fits.Header.fromtextfile(data_path('cubewcs1.hdr'))
    w1 = wcs.WCS(header)

    with pytest.raises(ValueError) as exc:
        newwcs = convert_spectral_axis(w1, 'km/s', 'VOPT')

    #assert exc.value.args[0] == ("There is no linear transformation from "
    #                             "FREQ to km / s")
    assert exc.value.args[0] == 'If converting from wavelength/frequency to speed, a reference wavelength/frequency is required.'

@pytest.mark.parametrize('wcstype',('Z','W','R','V'))
def test_greisen2006(wcstype):
    hdr = fits.Header.fromtextfile(data_path('greisen2006.hdr'))

    # We have not implemented frame conversions, so we can only convert bary
    # <-> bary in this case
    wcs0 = wcs.WCS(hdr, key='F')
    wcs1 = wcs.WCS(hdr, key=wcstype)

    if wcstype in ('R','V','Z'):
        if wcs1.wcs.restfrq:
            rest = wcs1.wcs.restfrq*u.Hz
        elif wcs1.wcs.restwav:
            rest = wcs1.wcs.restwav*u.m
    else:
        rest = None

    outunit = u.Unit(wcs1.wcs.cunit[wcs1.wcs.spec])
    out_ctype = wcs1.wcs.ctype[wcs1.wcs.spec]

    wcs2 = convert_spectral_axis(wcs0,
                                 outunit,
                                 out_ctype,
                                 rest_value=rest)
    np.testing.assert_almost_equal(wcs2.wcs.cdelt[wcs2.wcs.spec],
                                   wcs1.wcs.cdelt[wcs1.wcs.spec],
                                   decimal=2)
    np.testing.assert_almost_equal(wcs2.wcs.crval[wcs2.wcs.spec],
                                   wcs1.wcs.crval[wcs1.wcs.spec],
                                   decimal=2)
    assert wcs2.wcs.ctype[wcs2.wcs.spec] == wcs1.wcs.ctype[wcs1.wcs.spec]
    assert wcs2.wcs.cunit[wcs2.wcs.spec] == wcs1.wcs.cunit[wcs1.wcs.spec]

def test_byhand_f2v():
    # VELO-F2V
    CRVAL3F = 1.37847121643E+09
    CDELT3F = 9.764775E+04
    RESTFRQV= 1.420405752E+09
    CRVAL3V = 8.98134229811E+06
    CDELT3V = -2.1217551E+04
    CUNIT3V = 'm/s'
    CUNIT3F = 'Hz'

    crvalf = CRVAL3F * u.Unit(CUNIT3F)
    crvalv = CRVAL3V * u.Unit(CUNIT3V)
    restfreq = RESTFRQV * u.Unit(CUNIT3F)
    cdeltf = CDELT3F * u.Unit(CUNIT3F)
    cdeltv = CDELT3V * u.Unit(CUNIT3V)

    #crvalv_computed = crvalf.to(CUNIT3V, u.doppler_radio(restfreq))
    crvalv_computed = crvalf.to(CUNIT3V, u.doppler_relativistic(restfreq))
    cdeltv_computed = -4*constants.c*cdeltf*crvalf*restfreq**2 / (crvalf**2+restfreq**2)**2
    
    np.testing.assert_almost_equal(crvalv_computed, crvalv, decimal=3)
    np.testing.assert_almost_equal(cdeltv_computed, cdeltv, decimal=3)

def test_byhand_vrad():
    # VRAD
    CRVAL3F = 1.37847121643E+09
    CDELT3F = 9.764775E+04
    RESTFRQR= 1.420405752E+09
    CRVAL3R = 8.85075090419E+06
    CDELT3R = -2.0609645E+04
    CUNIT3R = 'm/s'
    CUNIT3F = 'Hz'

    crvalf = CRVAL3F * u.Unit(CUNIT3F)
    crvalv = CRVAL3R * u.Unit(CUNIT3R)
    restfreq = RESTFRQR * u.Unit(CUNIT3F)
    cdeltf = CDELT3F * u.Unit(CUNIT3F)
    cdeltv = CDELT3R * u.Unit(CUNIT3R)

    #crvalv_computed = crvalf.to(CUNIT3R, u.doppler_radio(restfreq))
    crvalv_computed = crvalf.to(CUNIT3R, u.doppler_radio(restfreq))
    cdeltv_computed = -(cdeltf / restfreq)*constants.c
    
    np.testing.assert_almost_equal(crvalv_computed, crvalv, decimal=3)
    np.testing.assert_almost_equal(cdeltv_computed, cdeltv, decimal=3)

def test_byhand_vopt():
    # VOPT: case "Z"
    CRVAL3F = 1.37847121643E+09
    CDELT3F = 9.764775E+04
    CUNIT3F = 'Hz'
    RESTWAVZ= 0.211061139
    #CTYPE3Z = 'VOPT-F2W'
    CRVAL3Z = 9.120000E+06
    CDELT3Z = -2.1882651E+04
    CUNIT3Z = 'm/s'

    crvalf = CRVAL3F * u.Unit(CUNIT3F)
    crvalv = CRVAL3Z * u.Unit(CUNIT3Z)
    restwav = RESTWAVZ * u.m
    cdeltf = CDELT3F * u.Unit(CUNIT3F)
    cdeltv = CDELT3Z * u.Unit(CUNIT3Z)

    #crvalv_computed = crvalf.to(CUNIT3R, u.doppler_radio(restwav))
    crvalw_computed = crvalf.to(u.m, u.spectral())
    cdeltw_computed = -(cdeltf / crvalf**2)*constants.c

    crvalv_computed = crvalw_computed.to(CUNIT3Z, u.doppler_optical(restwav))
    #cdeltv_computed = (cdeltw_computed *
    #                   4*constants.c*crvalw_computed*restwav**2 /
    #                   (restwav**2+crvalw_computed**2)**2)
    cdeltv_computed = (cdeltw_computed / restwav)*constants.c
    
    # Disagreement is 2.5e-7: good, but not really great...
    assert np.abs((crvalv_computed-crvalv)/crvalv) < 1e-6
    np.testing.assert_almost_equal(cdeltv_computed, cdeltv, decimal=2)
