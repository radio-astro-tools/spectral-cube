from astropy import wcs
from astropy import units as u

def _parse_velocity_convention(vc):
    if 'vc' in (u.doppler_radio, 'radio', 'RADIO', 'VRAD'):
        return u.doppler_radio
    elif 'vc' in (u.doppler_optical, 'optical', 'OPTICAL', 'VOPT'):
        return u.doppler_optical
    elif 'vc' in (u.doppler_relativistic, 'relativistic', 'RELATIVE', 'VREL'):
        raise ValueError("Only linear transformations are currently supported")
        return u.doppler_relativistic
    else:
        raise ValueError("Unrecognized velocity (doppler) convention")

def convert_spectral_axis(mywcs, outunit, velocity_convention=None):
    """
    Conversions between two equivalent LINEAR WCS systems
    """
    spwcs = mywcs.sub([wcs.WCSSUB_SPECTRAL])
    myunit = u.Unit(spwcs.wcs.cunit[0])
    refunit = mywcs.wcs.restfrq*u.Hz or mywcs.wcs.restwav*u.m

    if velocity_convention is None:
        # Assume the default / "natural" convention
        # Radio <-> freq
        # Optical <-> wavelength
        # Logic is to check *data* units, THEN reference, assuming
        # data wins in a conflict
        if u.Hz.is_equivalent(myunit):
            equiv = u.doppler_radio
        elif u.m.is_equivalent(myunit):
            equiv = u.doppler_optical
        elif u.Hz.is_equivalent(refunit):
            equiv = u.doppler_radio
        elif u.m.is_equivalent(refunit):
            equiv = u.doppler_optical
        else:
            raise ValueError("No velocity convention fits the WCS")
    else:
        equiv = _parse_velocity_convention(velocity_convention)

    crval = (spwcs.wcs.crval[0] * myunit)
    new_crval = crval.to(outunit, equiv(refunit))
    cdelt = (spwcs.wcs.cdelt[0] * myunit)
    delt1 = (crval + cdelt).to(outunit, equiv(refunit))
    new_cdelt = (delt1 - new_crval)

    new_wcs = spwcs.copy()

    new_wcs.wcs.crval[0] = new_crval.value
    new_wcs.wcs.cdelt[0] = new_cdelt.value

    return new_wcs

def join_wcs(wcs1, wcs2):
    """
    Merge two WCS objects
    """
    raise NotImplementedError("Waiting on https://github.com/astropy/astropy/issues/2361")
