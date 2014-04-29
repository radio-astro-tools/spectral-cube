from astropy import wcs
from astropy import units as u
from astropy import constants
import warnings

def _parse_velocity_convention(vc):
    if vc in (u.doppler_radio, 'radio', 'RADIO', 'VRAD', 'frequency'):
        return u.doppler_radio
    elif vc in (u.doppler_optical, 'optical', 'OPTICAL', 'VOPT', 'wavelength'):
        return u.doppler_optical
    elif vc in (u.doppler_relativistic, 'relativistic', 'RELATIVE', 'VREL', 'speed'):
        return u.doppler_relativistic
    else:
        raise ValueError("Unrecognized velocity (doppler) convention")

# These are the only linear transformations allowed
linear_ctypes = {u.doppler_optical: 'VOPT', u.doppler_radio: 'VRAD'}

all_ctypes = {'speed': linear_ctypes,
              'frequency': 'FREQ',
              'length': 'WAVE'}


def _get_linear_transformation(unit1, unit2):
    """
    Determin the default / "natural" convention
    Radio <-> freq
    Optical <-> wavelength
    """

    if unit1.is_equivalent(unit2):
        # Only a change of units is needed
        return lambda *args: []
    elif unit1.is_equivalent(unit2, u.spectral()):
        # wavelength <-> frequency
        # this is NOT a linear transformation (w = 1/v)
        return None
    elif (unit1.physical_type in ('frequency','speed') and
          unit2.physical_type in ('frequency','speed')):
        # top 'if' statement rules out both being the same
        return u.doppler_radio
    elif (unit1.physical_type in ('length','speed') and
          unit2.physical_type in ('length','speed')):
        return u.doppler_optical

def _get_transformation(unit1, unit2):
    pass

def convert_spectral_axis_linear(mywcs, outunit):
    """
    Conversions between two equivalent LINEAR WCS systems

    Does not support energy -> frequency, even though these are linear
    equivalent systems
    """
    spwcs = mywcs.sub([wcs.WCSSUB_SPECTRAL])
    myunit = u.Unit(spwcs.wcs.cunit[0])
    equiv = _get_linear_transformation(myunit, outunit)

    if equiv is None:
        raise ValueError("There is no linear transformation from "
                         "{0} to {1}".format(str(myunit),
                                             str(outunit)))

    ref_value = mywcs.wcs.restfrq*u.Hz or mywcs.wcs.restwav*u.m
    if equiv in (u.doppler_optical, u.doppler_radio) and ref_value.value == 0:
        raise ValueError("Must have a reference wavelength or frequency to "
                         "convert between velocity and wavelength or frequency.")

    crval = (spwcs.wcs.crval[0] * myunit)
    new_crval = crval.to(outunit, equiv(ref_value))
    cdelt = (spwcs.wcs.cdelt[0] * myunit)
    delt1 = (crval + cdelt).to(outunit, equiv(ref_value))
    new_cdelt = (delt1 - new_crval)

    new_wcs = mywcs.copy()

    new_wcs.wcs.crval[mywcs.wcs.spec] = new_crval.value
    new_wcs.wcs.cdelt[mywcs.wcs.spec] = new_cdelt.value
    new_wcs.wcs.ctype[mywcs.wcs.spec] = linear_ctypes[equiv]
    new_wcs.wcs.cunit[mywcs.wcs.spec] = str(outunit)

    return new_wcs

def convert_spectral_axis(mywcs, outunit, out_ctype):
    """

    Only VACUUM units are supported (not air)
    """
    myunit = u.Unit(mywcs.wcs.cunit[mywcs.wcs.spec])
    in_spec_ctype = mywcs.wcs.ctype[mywcs.wcs.spec]

    # Used to indicate the intial / final sampling system
    wcs_unit_dict = {'F': u.Hz, 'W': u.m, 'V': u.m/u.s}
    lin_ctype = (in_spec_ctype[5] if len(in_spec_ctype) > 4 else ' ') # 6th character
    lin_cunit = (wcs_unit_dict[lin_ctype] if lin_ctype in wcs_unit_dict
                 else mywcs.wcs.cunit[mywcs.wcs.spec])

    #in_linear = lin_ctype == ' '
    #out_linear = out_ctype[6] == ' '

    #if (in_linear and out_linear):
    #    return convert_spectral_axis_linear(mywcs, outunit)

    # The rest frequency and wavelength should be equivalent
    ref_value = mywcs.wcs.restfrq*u.Hz or mywcs.wcs.restwav*u.m

    # because of astropy magic, we can just convert directly from crval_in to crval_out
    # astropy takes care of the "chaining" described in Greisen 2006
    # the derivative is computed locally to the output unit
    crval_in = (mywcs.wcs.crval[mywcs.wcs.spec] * myunit)

    # Compute the X_r value, using eqns 6,8..16 in Greisein 2006
    # (the velocity conversions are all "apparent velocity", i.e. relativistic
    # convention)
    #crval_lin = crval_in.to(lin_cunit, u.doppler_relativistic(ref_value) + u.spectral())
    #crval_out = crval_lin.to(outunit,
    #                         _parse_velocity_convention(out_ctype[:4])(ref_value)
    #                         + u.spectral())

    # 1. Convert velocity to frequency or wavelength (or leave freq/wav alone)
    try:
        in_vcequiv = _parse_velocity_convention(lin_cunit.physical_type)(ref_value)
        crval_lin = crval_in.to(lin_cunit, in_vcequiv + u.spectral())
    except ValueError:
        crval_lin = crval_in.to(lin_cunit, u.spectral())

    # 2. Convert freq/wav to velocity (or wav/freq)
    try:
        vcequiv = _parse_velocity_convention(out_ctype[:4])(ref_value)
        crval_out = crval_lin.to(outunit, vcequiv + u.spectral())
    except ValueError:
        crval_out = crval_lin.to(outunit, u.spectral())

    # output cdelt is the derivative at the output CRVAL
    cdelt_in = (mywcs.wcs.cdelt[mywcs.wcs.spec] * myunit)
    cdelt_out = cdelt_derivative(crval_lin, cdelt_in,
                                 intype=myunit.physical_type,
                                 outtype=outunit.physical_type,
                                 rest=ref_value).to(outunit)

    # This block is if you wanted to *determine* out_ctype instead of specify it
    #nonlinearalgorithmcode = (typedict[myunit.physical_type] + "2" +
    #                          typedict[outunit.physical_type])
    #
    #if outunit.physical_type == 'speed':
    #    out_ctype = linear_ctypes[vcequiv]
    #else:
    #    out_ctype = all_ctypes[outunit.physical_type]

    #out_ctype += "_" + nonlinearalgorithmcode

    assert crval_out.unit == cdelt_out.unit

    newwcs = mywcs.deepcopy()
    newwcs.wcs.cdelt[newwcs.wcs.spec] = cdelt_out.value
    newwcs.wcs.cunit[newwcs.wcs.spec] = cdelt_out.unit.to_string(format='fits')
    newwcs.wcs.crval[newwcs.wcs.spec] = crval_out.value
    newwcs.wcs.ctype[newwcs.wcs.spec] = out_ctype

    return newwcs

def cdelt_derivative(crval, cdelt, intype, outtype, rest=None, equivalencies=[]):
    if set((outtype,intype)) == set(('wavelength','frequency')):
        # Symmetric equations!
        return -constants.c / crval**2 * cdelt
    elif outtype in ('frequency','wavelength') and intype == 'speed':
        numer = cdelt * constants.c * rest.to(cdelt.unit, u.spectral())
        denom = (constants.c + crval)*(constants.c**2 - crval**2)**0.5
        return numer / denom
    elif outtype == 'speed' and intype in ('frequency','wavelength'):
        numer = 4 * constants.c * crval * rest.to(crval.unit, u.spectral())**2 * cdelt
        denom = (crval**2 + rest.to(crval.unit)**2)**2
        if intype == 'frequency':
            return -numer/denom
        else:
            return numer/denom
    else:
        raise ValueError("Invalid in/out frames")
