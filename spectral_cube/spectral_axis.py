from astropy import wcs
from astropy import units as u
from astropy import constants
import warnings

def _parse_velocity_convention(vc):
    if vc in (u.doppler_radio, 'radio', 'RADIO', 'VRAD', 'F', 'FREQ'):
        return u.doppler_radio
    elif vc in (u.doppler_optical, 'optical', 'OPTICAL', 'VOPT', 'W', 'WAVE'):
        return u.doppler_optical
    elif vc in (u.doppler_relativistic, 'relativistic', 'RELATIVE', 'VREL', 'speed', 'V', 'VELO'):
        return u.doppler_relativistic
    #else:
    #    raise ValueError("Unrecognized velocity (doppler) convention")

# These are the only linear transformations allowed
linear_ctypes = {u.doppler_optical: 'VOPT', u.doppler_radio: 'VRAD',
                 u.doppler_relativistic: 'VELO'}
linear_ctype_chars = {u.doppler_optical: 'W', u.doppler_radio: 'F',
                      u.doppler_relativistic: 'V'}

all_ctypes = {'speed': linear_ctypes,
              'frequency': 'FREQ',
              'length': 'WAVE'}

ctype_to_physicaltype = {'W': 'length',
                         'A': 'air wavelength', # unsupported
                         'F': 'frequency',
                         'V': 'speed'}

physical_type_to_ctype = dict([(v,k) for k,v in ctype_to_physicaltype.iteritems()])


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

def determine_ctype_from_vconv(ctype, unit, velocity_convention=None):
    """
    Given a CTYPE describing the current WCS and an output unit and velocity
    convention, determine the appropriate output CTYPE

    Examples
    --------
    >>> determine_ctype_from_vconv('VELO-F2V', u.Hz)
    'FREQ'
    >>> determine_ctype_from_vconv('VELO-F2V', u.m)
    'WAVE-F2W'
    >>> determine_ctype_from_vconv('FREQ', u.m/u.s)
    ValueError('A velocity convention must be specified')
    >>> determine_ctype_from_vconv('FREQ', u.m/u.s, velocity_convention=u.doppler_radio)
    'VRAD'
    >>> determine_ctype_from_vconv('FREQ', u.m/u.s, velocity_convention=u.doppler_optical)
    'VOPT-F2W'
    >>> determine_ctype_from_vconv('FREQ', u.m/u.s, velocity_convention=u.doppler_relativistic)
    'VELO-F2V'
    """
    unit = u.Unit(unit)

    in_physchar = (ctype[0]if len(ctype)<=4 else
                   ctype[5])

    if unit.physical_type == 'speed':
        if velocity_convention is None:
            raise ValueError('A velocity convention must be specified')
        vcin = _parse_velocity_convention(ctype[:4])
        vcout = _parse_velocity_convention(velocity_convention)
        if vcin == vcout:
            return linear_ctypes[vcout]
        else:
            return "{type}-{s1}2{s2}".format(type=linear_ctypes[vcout],
                                             s1=in_physchar,
                                             s2=linear_ctype_chars[vcout])
            
    else:
        in_phystype = ctype_to_physicaltype[in_physchar]
        if in_phystype == unit.physical_type:
            # Linear case
            return all_ctypes[in_phystype]
        else:
            # Nonlinear case
            out_physchar = physical_type_to_ctype[unit.physical_type]
            return "{type}-{s1}2{s2}".format(type=all_ctypes[unit.physical_type],
                                             s1=in_physchar,
                                             s2=out_physchar)



def get_restfreq_from_wcs(mywcs):
    if mywcs.wcs.restfrq:
        ref_value = mywcs.wcs.restfrq*u.Hz
        return ref_value
    elif mywcs.wcs.restwav:
        ref_value = mywcs.wcs.restwav*u.m
        return ref_value

def convert_spectral_axis_linear(mywcs, outunit, rest_value=None):
    """
    Conversions between two equivalent LINEAR WCS systems

    Does not support energy -> frequency, even though these are linear
    equivalent systems
    """
    spwcs = mywcs.sub([wcs.WCSSUB_SPECTRAL])
    myunit = u.Unit(spwcs.wcs.cunit[0])
    outunit = u.Unit(outunit)
    equiv = _get_linear_transformation(myunit, outunit)

    if equiv is None:
        raise ValueError("There is no linear transformation from "
                         "{0} to {1}".format(str(myunit),
                                             str(outunit)))

    if outunit.physical_type == 'speed':
        if rest_value is None:
            ref_value = get_restfreq_from_wcs(mywcs)
            if ref_value is None:
                raise ValueError("If converting from wavelength/frequency to speed, "
                                 "a reference wavelength/frequency is required.")
        else:
            ref_value = rest_value.to(u.Hz, u.spectral())
    elif myunit.physical_type == 'speed':
        # The rest frequency and wavelength should be equivalent
        ref_value = get_restfreq_from_wcs(mywcs)
        if ref_value is None:
            raise ValueError("If converting from speed to wavelength/frequency, "
                             "a reference wavelength/frequency is required.")

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

def convert_spectral_axis(mywcs, outunit, out_ctype, rest_value=None):
    """

    Only VACUUM units are supported (not air)
    """
    outunit = u.Unit(outunit)
    myunit = u.Unit(mywcs.wcs.cunit[mywcs.wcs.spec])
    in_spec_ctype = mywcs.wcs.ctype[mywcs.wcs.spec]

    # Used to indicate the intial / final sampling system
    wcs_unit_dict = {'F': u.Hz, 'W': u.m, 'V': u.m/u.s}
    lin_ctype = (in_spec_ctype[5] if len(in_spec_ctype) > 4 else ' ') # 6th character
    lin_cunit = (wcs_unit_dict[lin_ctype] if lin_ctype in wcs_unit_dict
                 else mywcs.wcs.cunit[mywcs.wcs.spec])
    if lin_ctype == ' ':
        lin_ctype = in_spec_ctype[:4]

    out_ctype_conv = out_ctype[7] if len(out_ctype) > 4 else out_ctype[:4]
    vcequiv = _parse_velocity_convention(out_ctype_conv)

    if ((_get_linear_transformation(myunit, outunit) == vcequiv)):
        return convert_spectral_axis_linear(mywcs, outunit, rest_value=rest_value)

    ref_value = None
    if outunit.physical_type == 'speed':
        if rest_value is None:
            raise ValueError("If converting from wavelength/frequency to speed, "
                             "a reference wavelength/frequency is required.")
        ref_value = rest_value.to(u.Hz, u.spectral())
    elif myunit.physical_type == 'speed':
        # The rest frequency and wavelength should be equivalent
        if mywcs.wcs.restfrq:
            ref_value = mywcs.wcs.restfrq*u.Hz
        elif mywcs.wcs.restwav:
            ref_value = mywcs.wcs.restwav*u.m
        else:
            raise ValueError("If converting from speed to wavelength/frequency, "
                             "a reference wavelength/frequency is required.")

    # because of astropy magic, we can just convert directly from crval_in to crval_out
    # astropy takes care of the "chaining" described in Greisen 2006
    # the derivative is computed locally to the output unit
    crval_in = (mywcs.wcs.crval[mywcs.wcs.spec] * myunit)

    # Compute the X_r value, using eqns 6,8..16 in Greisein 2006
    # (the velocity conversions are all "apparent velocity", i.e. relativistic
    # convention)
    # The "_lin" things refer to X_r coordinates, i.e. these are the base unit
    # from which velocities would be considered to be linear...

    # 1. Convert velocity to frequency or wavelength (or leave freq/wav alone)
    #try:
    #    in_vcequiv = _parse_velocity_convention(lin_ctype)(ref_value)
    #    crval_lin = crval_in.to(lin_cunit, in_vcequiv + u.spectral())
    #except ValueError:
    #    crval_lin = crval_in.to(lin_cunit, u.spectral())
    if ref_value is None:
        crval_lin = crval_in.to(lin_cunit, u.spectral())
    else:
        crval_lin = crval_in.to(lin_cunit, u.spectral() +
                                u.doppler_relativistic(ref_value))

    # output cdelt is the derivative at the output CRVAL
    # Compute the "linear" version of cdelt... (i.e., the quantity that CDELT
    # will be linearly proportional to)
    cdelt_in = (mywcs.wcs.cdelt[mywcs.wcs.spec] * myunit)
    cdelt_lin = cdelt_derivative(crval_lin, cdelt_in,
                                 intype=myunit.physical_type,
                                 outtype=ctype_to_physicaltype[out_ctype_conv],
                                 rest=ref_value)
    # 2. Convert freq/wav to velocity (or wav/freq)
    if vcequiv is not None and ref_value is not None:
        crval_out = crval_lin.to(outunit, vcequiv(ref_value) + u.spectral())

        if cdelt_lin.unit.physical_type == 'speed':
            cdelt_out = cdelt_lin.to(outunit, vcequiv(ref_value) + u.spectral())
        else:
            # this little bit of apparent redundancy is because there is no "delta"
            # equivalency, and defining one is tedious
            # It is easier to recognize that cdelt+crval can be converted...
            val_to_get_cdelt = (cdelt_lin+ref_value.to(cdelt_lin.unit, u.spectral()))
            cdelt_out = val_to_get_cdelt.to(outunit, vcequiv(ref_value) + u.spectral())
    else:
        crval_out = crval_lin.to(outunit, u.spectral())
        cdelt_out = cdelt_lin.to(outunit, u.spectral())

    if crval_out.unit != cdelt_out.unit:
        # this should not be possible, but it's a sanity check
        raise ValueError("Conversion failed: the units of cdelt and crval don't match.")

    # A cdelt of 0 would be meaningless
    if cdelt_out.value == 0:
        raise ValueError("Conversion failed: the output CDELT would be 0.")

    newwcs = mywcs.deepcopy()
    newwcs.wcs.cdelt[newwcs.wcs.spec] = cdelt_out.value
    newwcs.wcs.cunit[newwcs.wcs.spec] = cdelt_out.unit.to_string(format='fits')
    newwcs.wcs.crval[newwcs.wcs.spec] = crval_out.value
    newwcs.wcs.ctype[newwcs.wcs.spec] = out_ctype

    return newwcs

def cdelt_derivative(crval, cdelt, intype, outtype, rest=None, equivalencies=[],
                     convention=None):
    if set((outtype,intype)) == set(('length','frequency')):
        # Symmetric equations!
        return -constants.c / crval**2 * cdelt
    elif outtype in ('frequency','length') and intype == 'speed':
        numer = cdelt * constants.c * rest.to(cdelt.unit, u.spectral())
        denom = (constants.c + crval)*(constants.c**2 - crval**2)**0.5
        return numer / denom
    elif outtype == 'speed' and intype in ('frequency','length'):
        numer = 4 * constants.c * crval * rest.to(crval.unit, u.spectral())**2 * cdelt
        denom = (crval**2 + rest.to(crval.unit)**2)**2
        if intype == 'frequency':
            return -numer/denom
        else:
            return numer/denom
    else:
        raise ValueError("Invalid in/out frames")
