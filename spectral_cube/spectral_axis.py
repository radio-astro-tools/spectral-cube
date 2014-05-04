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

# These are the only linear transformations allowed
LINEAR_CTYPES = {u.doppler_optical: 'VOPT', u.doppler_radio: 'VRAD',
                 u.doppler_relativistic: 'VELO'}
LINEAR_CTYPE_CHARS = {u.doppler_optical: 'W', u.doppler_radio: 'F',
                      u.doppler_relativistic: 'V'}

ALL_CTYPES = {'speed': LINEAR_CTYPES,
              'frequency': 'FREQ',
              'length': 'WAVE'}

CTYPE_TO_PHYSICALTYPE = {'WAVE': 'length',
                         'AIR': 'air wavelength', # unsupported
                         'FREQ': 'frequency',
                         'VELO': 'speed',
                         'VRAD': 'speed',
                         'VOPT': 'speed',
                         }

CTYPE_CHAR_TO_PHYSICALTYPE = {'W': 'length',
                              'A': 'air wavelength', # unsupported
                              'F': 'frequency',
                              'V': 'speed'}

CTYPE_TO_PHYSICALTYPE.update(CTYPE_CHAR_TO_PHYSICALTYPE)

PHYSICAL_TYPE_TO_CTYPE = dict([(v,k) for k,v in
                               CTYPE_CHAR_TO_PHYSICALTYPE.iteritems()])

# Used to indicate the intial / final sampling system
WCS_UNIT_DICT = {'F': u.Hz, 'W': u.m, 'V': u.m/u.s}
PHYS_UNIT_DICT = {'length': u.m, 'frequency': u.Hz, 'speed': u.m/u.s}

LINEAR_CUNIT_DICT = {'VRAD': u.Hz, 'VOPT': u.m}
LINEAR_CUNIT_DICT.update(WCS_UNIT_DICT)

def _get_linear_equivalency(unit1, unit2):
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
            return LINEAR_CTYPES[vcout]
        else:
            return "{type}-{s1}2{s2}".format(type=LINEAR_CTYPES[vcout],
                                             s1=in_physchar,
                                             s2=LINEAR_CTYPE_CHARS[vcout])
            
    else:
        in_phystype = CTYPE_CHAR_TO_PHYSICALTYPE[in_physchar]
        if in_phystype == unit.physical_type:
            # Linear case
            return ALL_CTYPES[in_phystype]
        else:
            # Nonlinear case
            out_physchar = PHYSICAL_TYPE_TO_CTYPE[unit.physical_type]
            return "{type}-{s1}2{s2}".format(type=ALL_CTYPES[unit.physical_type],
                                             s1=in_physchar,
                                             s2=out_physchar)



def get_rest_value_from_wcs(mywcs):
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
    equiv = _get_linear_equivalency(myunit, outunit)

    if equiv is None:
        raise ValueError("There is no linear transformation from "
                         "{0} to {1}".format(str(myunit),
                                             str(outunit)))

    if outunit.physical_type == 'speed':
        if rest_value is None:
            ref_value = get_rest_value_from_wcs(mywcs)
            if ref_value is None:
                raise ValueError("If converting from wavelength/frequency to speed, "
                                 "a reference wavelength/frequency is required.")
        else:
            ref_value = rest_value.to(u.Hz, u.spectral())
    elif myunit.physical_type == 'speed':
        # The rest frequency and wavelength should be equivalent
        ref_value = get_rest_value_from_wcs(mywcs)
        if ref_value is None:
            raise ValueError("If converting from speed to wavelength/frequency, "
                             "a reference wavelength/frequency is required.")

    crval = (spwcs.wcs.crval[0] * myunit)
    new_crval = crval.to(outunit, equiv(ref_value))
    cdelt = (spwcs.wcs.cdelt[0] * myunit)
    delt1 = (crval + cdelt).to(outunit, equiv(ref_value))
    new_cdelt = (delt1 - new_crval)

    new_wcs = mywcs.deepcopy()

    new_wcs.wcs.crval[mywcs.wcs.spec] = new_crval.value
    new_wcs.wcs.cdelt[mywcs.wcs.spec] = new_cdelt.value
    new_wcs.wcs.ctype[mywcs.wcs.spec] = LINEAR_CTYPES[equiv]
    new_wcs.wcs.cunit[mywcs.wcs.spec] = str(outunit)

    return new_wcs

def convert_spectral_axis(mywcs, outunit, out_ctype, rest_value=None, debug=False):
    """

    Only VACUUM units are supported (not air)

    Process:
        1. Convert the input unit to its equivalent linear unit
        2. Convert the input linear unit to the output linear unit
        3. Convert the output linear unit to the output unit
    """
    outunit = u.Unit(outunit)
    myunit = u.Unit(mywcs.wcs.cunit[mywcs.wcs.spec])
    in_spec_ctype = mywcs.wcs.ctype[mywcs.wcs.spec]

    # If the input unit is not linearly sampled, its linear equivalent will be
    # the 8th character in the ctype, and the linearly-sampled ctype will be
    # the 6th character
    lin_ctype = (in_spec_ctype[7] if len(in_spec_ctype) > 4 else in_spec_ctype[:4])
    lin_cunit = (LINEAR_CUNIT_DICT[lin_ctype] if lin_ctype in LINEAR_CUNIT_DICT
                 else mywcs.wcs.cunit[mywcs.wcs.spec])

    in_vcequiv = _parse_velocity_convention(in_spec_ctype[:4])

    out_ctype_conv = out_ctype[7] if len(out_ctype) > 4 else out_ctype[:4]
    out_lin_cunit = (LINEAR_CUNIT_DICT[out_ctype_conv] if out_ctype_conv in
                     LINEAR_CUNIT_DICT else outunit)
    vcequiv = _parse_velocity_convention(out_ctype_conv)

    #if ((_get_linear_equivalency(myunit, outunit) == vcequiv)):
    #    return convert_spectral_axis_linear(mywcs, outunit, rest_value=rest_value)

    ref_value = None
    if outunit.physical_type == 'speed':
        if rest_value is None:
            rest_value = get_rest_value_from_wcs(mywcs)
            if rest_value is None:
                raise ValueError("If converting from wavelength/frequency to speed, "
                                 "a reference wavelength/frequency is required.")
            else:
                warnings.warn("Using WCS built-in rest frequency even though the "
                              "WCS system was originally {0}".format(in_spec_ctype))
        ref_value = rest_value.to(u.Hz, u.spectral())
    elif myunit.physical_type == 'speed':
        # The rest frequency and wavelength should be equivalent
        wcs_rv = get_rest_value_from_wcs(mywcs)
        if rest_value is not None:
            warnings.warn("Overriding the reference value specified in the WCS.")
            ref_value = rest_value
        elif wcs_rv is not None:
            ref_value = wcs_rv
        else:
            raise ValueError("If converting from speed to wavelength/frequency, "
                             "a reference wavelength/frequency is required.")

    # Load the input values
    crval_in = (mywcs.wcs.crval[mywcs.wcs.spec] * myunit)
    cdelt_in = (mywcs.wcs.cdelt[mywcs.wcs.spec] * myunit)

    # Compute the X_r value, using eqns 6,8..16 in Greisen 2006
    # (the velocity conversions are all "apparent velocity", i.e. relativistic
    # convention)
    # The "_lin" things refer to X_r coordinates, i.e. these are the base unit
    # from which velocities would be considered to be linear...

    # 1. Convert input to input, linear
    if in_vcequiv is not None and ref_value is not None:
        crval_lin1 = crval_in.to(lin_cunit, u.spectral() + in_vcequiv(ref_value))
    else:
        crval_lin1 = crval_in.to(lin_cunit, u.spectral())
    cdelt_lin1 = cdelt_derivative(crval_in,
                                  cdelt_in,
                                  # equivalent: myunit.physical_type
                                  intype=CTYPE_TO_PHYSICALTYPE[in_spec_ctype[:4]],
                                  outtype=CTYPE_TO_PHYSICALTYPE[lin_ctype],
                                  rest=ref_value,
                                  linear=True
                                  )

    # 2. Convert input, linear to output, linear
    if ref_value is None:
        if in_vcequiv is not None:
            pass # consider raising a ValueError here; not clear if this is valid
        crval_lin2 = crval_lin1.to(out_lin_cunit, u.spectral())
    else:
        # at this stage, the transition can ONLY be relativistic, because the V
        # frame (as a linear frame) is only defined as "apparent velocity"
        crval_lin2 = crval_lin1.to(out_lin_cunit, u.spectral() +
                                   u.doppler_relativistic(ref_value))

    # For cases like VRAD <-> FREQ and VOPT <-> WAVE, this will be linear too:
    linear_middle = in_vcequiv == vcequiv

    cdelt_lin2 = cdelt_derivative(crval_lin1, cdelt_lin1,
                                  intype=CTYPE_TO_PHYSICALTYPE[lin_ctype],
                                  outtype=CTYPE_TO_PHYSICALTYPE[out_ctype_conv],
                                  rest=ref_value,
                                  linear=linear_middle)

    # 3. Convert output, linear to output
    if vcequiv is not None and ref_value is not None:
        crval_out = crval_lin2.to(outunit, vcequiv(ref_value) + u.spectral())
        #cdelt_out = cdelt_lin2.to(outunit, vcequiv(ref_value) + u.spectral())
        cdelt_out = cdelt_derivative(crval_lin2,
                                     cdelt_lin2,
                                     intype=CTYPE_TO_PHYSICALTYPE[out_ctype_conv],
                                     outtype=outunit.physical_type,
                                     rest=ref_value,
                                     linear=True
                                     ).to(outunit)
    else:
        crval_out = crval_lin2.to(outunit, u.spectral())
        cdelt_out = cdelt_lin2.to(outunit, u.spectral())

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
    if rest_value is not None:
        if rest_value.unit.physical_type == 'frequency':
            newwcs.wcs.restfrq = rest_value.to(u.Hz).value
        elif rest_value.unit.physical_type == 'length':
            newwcs.wcs.restwav = rest_value.to(u.m).value
        else:
            raise ValueError("Rest Value was specified, but not in frequency or length units")

    if debug:
        import pdb; pdb.set_trace()

    return newwcs

def cdelt_derivative(crval, cdelt, intype, outtype, linear=False, rest=None):
    if intype == outtype:
        return cdelt
    elif set((outtype,intype)) == set(('length','frequency')):
        # Symmetric equations!
        return (-constants.c / crval**2 * cdelt).to(PHYS_UNIT_DICT[outtype])
    elif outtype in ('frequency','length') and intype == 'speed':
        if linear:
            numer = cdelt * rest.to(PHYS_UNIT_DICT[outtype], u.spectral())
            denom = constants.c
        else:
            numer = cdelt * constants.c * rest.to(PHYS_UNIT_DICT[outtype], u.spectral())
            denom = (constants.c + crval)*(constants.c**2 - crval**2)**0.5
        if outtype == 'frequency':
            return (-numer/denom).to(PHYS_UNIT_DICT[outtype], u.spectral())
        else:
            return (numer/denom).to(PHYS_UNIT_DICT[outtype], u.spectral())
    elif outtype == 'speed' and intype in ('frequency','length'):

        if linear:
            numer = cdelt * constants.c
            denom = rest.to(PHYS_UNIT_DICT[intype], u.spectral())
        else:
            numer = 4 * constants.c * crval * rest.to(crval.unit, u.spectral())**2 * cdelt
            denom = (crval**2 + rest.to(crval.unit, u.spectral())**2)**2
        if intype == 'frequency':
            return (-numer/denom).to(PHYS_UNIT_DICT[outtype], u.spectral())
        else:
            return (numer/denom).to(PHYS_UNIT_DICT[outtype], u.spectral())
    else:
        raise ValueError("Invalid in/out frames")
