from __future__ import print_function, absolute_import, division

import six
import numpy as np
import struct
import warnings
import string
from astropy import log
from astropy.io import registry as io_registry
from ..spectral_cube import BaseSpectralCube
from .fits import load_fits_cube

"""
.. TODO::
    When any section length is zero, that means the following values are to be
    ignored.  No warning is needed.
"""

# Constant:
r2deg = 180/np.pi

# see sicfits.f90
_ctype_dict={'LII':'GLON',
             'BII':'GLAT',
             'VELOCITY':'VELO',
             'RA':'RA',
             'DEC':'DEC',
             'FREQUENCY': 'FREQ',
            }
_cunit_dict = {'LII':'deg',
               'BII':'deg',
               'VELOCITY':'km s-1',
               'RA':'deg',
               'DEC':'deg',
               'FREQUENCY': 'MHz',
              }
cel_types = ('RA','DEC','GLON','GLAT')

# CLASS apparently defaults to an ARC (zenithal equidistant) projection; this
# is what is output in case the projection # is zero when exporting from CLASS
_proj_dict = {0:'ARC', 1:'TAN', 2:'SIN', 3:'AZP', 4:'STG', 5:'ZEA', 6:'AIT',
              7:'GLS', 8:'SFL', }
_bunit_dict = {'k (tmb)': 'K'}

def is_lmv(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether input is in GILDAS CLASS lmv format
    """
    return filepath is not None and filepath.lower().endswith('.lmv')

def read_lmv(lf):
    """
    Read an LMV cube file

    Specification is primarily in GILDAS image_def.f90
    """
    log.warning("CLASS LMV cube reading is tentatively supported.  "
             "Please post bug reports at the first sign of danger!")

    # lf for "LMV File"
    filetype = _read_string(lf, 12)
    #!---------------------------------------------------------------------
    #! @ private
    #!       SYCODE system code
    #!       '-'    IEEE
    #!       '.'    EEEI (IBM like)
    #!       '_'    VAX
    #!       IMCODE file code
    #!       '<'    IEEE  64 bits    (Little Endian, 99.9 % of recent computers)
    #!       '>'    EEEI  64 bits    (Big Endian, HPUX, IBM-RISC, and SPARC ...)
    #!---------------------------------------------------------------------
    imcode = filetype[6]
    if filetype[:6] != 'GILDAS' or filetype[7:] != 'IMAGE':
        raise TypeError("File is not a GILDAS Image file")

    if imcode in ('<','>'):
        if imcode =='>':
            log.warning("Swap the endianness first...")
        return read_lmv_type2(lf)
    else:
        return read_lmv_type1(lf)

def read_lmv_type1(lf):
    header = {}
    # fmt probably matters!  Default is "r4", i.e. float32 data, but could be float64
    fmt = np.fromfile(lf, dtype='int32', count=1) # 4
    # number of data blocks
    ndb = np.fromfile(lf, dtype='int32', count=1) # 5
    gdf_type = np.fromfile(lf, dtype='int32', count=1) # 6
    # Reserved Space
    reserved_fill = np.fromfile(lf, dtype='int32', count=4) # 7
    general_section_length = np.fromfile(lf, dtype='int32', count=1) # 11
    #print "Format: ",fmt," ndb: ",ndb, " fill: ",fill," other: ",unknown

    # pos 12
    naxis,naxis1,naxis2,naxis3,naxis4 = np.fromfile(lf,count=5,dtype='int32')
    header['NAXIS'] = naxis
    header['NAXIS1'] = naxis1
    header['NAXIS2'] = naxis2
    header['NAXIS3'] = naxis3
    header['NAXIS4'] = naxis4

    # We are indexing bytes from here; CLASS indices are higher by 12
    # pos 17
    header['CRPIX1'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CRVAL1'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CDELT1'] = np.fromfile(lf,count=1,dtype='float64')[0] * r2deg
    header['CRPIX2'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CRVAL2'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CDELT2'] = np.fromfile(lf,count=1,dtype='float64')[0] * r2deg
    header['CRPIX3'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CRVAL3'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CDELT3'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CRPIX4'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CRVAL4'] = np.fromfile(lf,count=1,dtype='float64')[0]
    header['CDELT4'] = np.fromfile(lf,count=1,dtype='float64')[0]
    # pos 41
    #print "Post-crval",lf.tell()
    blank_section_length = np.fromfile(lf,count=1,dtype='int32')
    if blank_section_length != 8:
        warnings.warn("Invalid section length found for blanking section")
    bval = np.fromfile(lf,count=1,dtype='float32')[0] # 42
    header['TOLERANC'] = np.fromfile(lf,count=1,dtype='int32')[0] # 43 eval = tolerance
    extrema_section_length = np.fromfile(lf,count=1,dtype='int32')[0] # 44
    if extrema_section_length != 40:
        warnings.warn("Invalid section length found for extrema section")
    vmin,vmax = np.fromfile(lf,count=2,dtype='float32') # 45
    xmin,xmax,ymin,ymax,zmin,zmax = np.fromfile(lf,count=6,dtype='int32') # 47
    wmin,wmax = np.fromfile(lf,count=2,dtype='int32') # 53
    description_section_length = np.fromfile(lf,count=1,dtype='int32')[0] # 55
    if description_section_length != 72:
        warnings.warn("Invalid section length found for description section")
    #strings = lf.read(description_section_length) # 56
    header['BUNIT']  = _read_string(lf, 12) # 56
    header['CTYPE1'] = _read_string(lf, 12) # 59
    header['CTYPE2'] = _read_string(lf, 12) # 62
    header['CTYPE3'] = _read_string(lf, 12) # 65
    header['CTYPE4'] = _read_string(lf, 12) # 68
    header['CUNIT1'] = _cunit_dict[header['CTYPE1'].strip()]
    header['CUNIT2'] = _cunit_dict[header['CTYPE2'].strip()]
    header['CUNIT3'] = _cunit_dict[header['CTYPE3'].strip()]
    header['COOSYS'] = _read_string(lf, 12) # 71
    position_section_length = np.fromfile(lf,count=1,dtype='int32') # 74
    if position_section_length != 48:
        warnings.warn("Invalid section length found for position section")
    header['OBJNAME'] = _read_string(lf, 4*3) # 75
    header['RA'] = np.fromfile(lf, count=1, dtype='float64')[0] * r2deg # 78
    header['DEC'] = np.fromfile(lf, count=1, dtype='float64')[0] * r2deg # 80
    header['GLON'] = np.fromfile(lf, count=1, dtype='float64')[0] * r2deg # 82
    header['GLAT'] = np.fromfile(lf, count=1, dtype='float64')[0] * r2deg # 84
    header['EQUINOX'] = np.fromfile(lf,count=1,dtype='float32')[0] # 86
    header['PROJWORD'] = _read_string(lf, 4) # 87
    header['PTYP'] = np.fromfile(lf,count=1,dtype='int32')[0] # 88
    header['A0'] = np.fromfile(lf,count=1,dtype='float64')[0] # 89
    header['D0'] = np.fromfile(lf,count=1,dtype='float64')[0] # 91
    header['PANG'] = np.fromfile(lf,count=1,dtype='float64')[0] # 93
    header['XAXI'] = np.fromfile(lf,count=1,dtype='float32')[0] # 95
    header['YAXI'] = np.fromfile(lf,count=1,dtype='float32')[0] # 96
    spectroscopy_section_length = np.fromfile(lf,count=1,dtype='int32') # 97
    if spectroscopy_section_length != 48:
        warnings.warn("Invalid section length found for spectroscopy section")
    header['RECVR'] = _read_string(lf, 12) # 98
    header['FRES'] = np.fromfile(lf,count=1,dtype='float64')[0] # 101
    header['IMAGFREQ'] = np.fromfile(lf,count=1,dtype='float64')[0] # 103 "FIMA"
    header['REFFREQ'] = np.fromfile(lf,count=1,dtype='float64')[0] # 105
    header['VRES'] = np.fromfile(lf,count=1,dtype='float32')[0] # 107
    header['VOFF'] = np.fromfile(lf,count=1,dtype='float32')[0] # 108
    header['FAXI'] = np.fromfile(lf,count=1,dtype='int32')[0] # 109
    resolution_section_length = np.fromfile(lf,count=1,dtype='int32')[0] # 110
    if resolution_section_length != 12:
        warnings.warn("Invalid section length found for resolution section")
    #header['DOPP'] = np.fromfile(lf,count=1,dtype='float16')[0] # 110a ???
    #header['VTYP'] = np.fromfile(lf,count=1,dtype='int16')[0] # 110b
    # integer, parameter :: vel_unk = 0      ! Unsupported referential :: planetary...)
    # integer, parameter :: vel_lsr = 1      ! LSR referential
    # integer, parameter :: vel_hel = 2      ! Heliocentric referential
    # integer, parameter :: vel_obs = 3      ! Observatory referential
    # integer, parameter :: vel_ear = 4      ! Earth-Moon barycenter referential
    # integer, parameter :: vel_aut = -1     ! Take referential from data
    header['BMAJ'] = np.fromfile(lf,count=1,dtype='float32')[0] # 111
    header['BMIN'] = np.fromfile(lf,count=1,dtype='float32')[0] # 112
    header['BPA'] = np.fromfile(lf,count=1,dtype='float32')[0] # 113
    noise_section_length = np.fromfile(lf,count=1,dtype='int32')
    if noise_section_length != 0:
        warnings.warn("Invalid section length found for noise section")
    header['NOISE'] = np.fromfile(lf,count=1,dtype='float32')[0] # 115
    header['RMS'] = np.fromfile(lf,count=1,dtype='float32')[0] # 116
    astrometry_section_length = np.fromfile(lf,count=1,dtype='int32')
    if astrometry_section_length != 0:
        warnings.warn("Invalid section length found for astrometry section")
    header['MURA'] = np.fromfile(lf,count=1,dtype='float32')[0] # 118
    header['MUDEC'] = np.fromfile(lf,count=1,dtype='float32')[0] # 119
    header['PARALLAX'] = np.fromfile(lf,count=1,dtype='float32')[0] # 120

    # Apparently CLASS headers aren't required to fill the 'value at
    # reference pixel' column
    if (header['CTYPE1'].strip() == 'RA' and header['CRVAL1'] == 0 and
        header['RA'] != 0):
        header['CRVAL1'] = header['RA']
        header['CRVAL2'] = header['DEC']


    # Copied from the type 2 reader:
    # Use the appropriate projection type
    ptyp = header['PTYP']
    for kw in header:
        if 'CTYPE' in kw:
            if header[kw].strip() in cel_types:
                n_dashes = 5-len(header[kw].strip())
                header[kw] = header[kw].strip()+ '-'*n_dashes + _proj_dict[ptyp]

    other_info = np.fromfile(lf, count=7, dtype='float32') # 121-end
    if not np.all(other_info == 0):
        warnings.warn("Found additional information in the last 7 bytes")

    endpoint = 508
    if lf.tell() != endpoint:
        raise ValueError("Header was not parsed correctly")

    data = np.fromfile(lf, count=naxis1*naxis2*naxis3, dtype='float32')

    data[data == bval] = np.nan

    # for no apparent reason, y and z are 1-indexed and x is zero-indexed
    if (wmin-1,zmin-1,ymin-1,xmin) != np.unravel_index(np.nanargmin(data),
                                                       [naxis4,naxis3,naxis2,naxis1]):
        warnings.warn("Data min location does not match that on file.  "
                      "Possible error reading data.")
    if (wmax-1,zmax-1,ymax-1,xmax) != np.unravel_index(np.nanargmax(data),
                                                       [naxis4,naxis3,naxis2,naxis1]):
        warnings.warn("Data max location does not match that on file.  "
                      "Possible error reading data.")
    if np.nanmax(data) != vmax:
        warnings.warn("Data max does not match that on file.  "
                      "Possible error reading data.")
    if np.nanmin(data) != vmin:
        warnings.warn("Data min does not match that on file.  "
                      "Possible error reading data.")

    return data.reshape([naxis4,naxis3,naxis2,naxis1]),header
    # debug
    #return data.reshape([naxis3,naxis2,naxis1]), header, hdr_f, hdr_s, hdr_i, hdr_d, hdr_d_2

def read_lmv_tofits(fileobj):
    from astropy.io import fits
    data,header = read_lmv(fileobj)
    # LMV may contain extra dimensions that are improperly labeled
    data = data.squeeze()
    bad_kws = ['NAXIS4','CRVAL4','CRPIX4','CDELT4','CROTA4','CUNIT4','CTYPE4']

    cards = [fits.header.Card(keyword=k, value=v[0], comment=v[1])
             if isinstance(v, tuple) else
             fits.header.Card(''.join(s for s in k if s in string.printable),
                              ''.join(s for s in v if s in string.printable)
                              if isinstance(v, six.string_types) else v)
             for k,v in six.iteritems(header)
             if k not in bad_kws]
    Header = fits.Header(cards)
    hdu = fits.PrimaryHDU(data=data, header=Header)
    return hdu

def load_lmv_cube(fileobj, target_cls=None, use_dask=None):
    hdu = read_lmv_tofits(fileobj)
    meta = {'filename':fileobj.name}

    return load_fits_cube(hdu, meta=meta, use_dask=use_dask)

def _read_byte(f):
    '''Read a single byte (from idlsave)'''
    return np.uint8(struct.unpack('=B', f.read(4)[:1])[0])

def _read_int16(f):
    '''Read a signed 16-bit integer (from idlsave)'''
    return np.int16(struct.unpack('=h', f.read(4)[2:4])[0])

def _read_int32(f):
    '''Read a signed 32-bit integer (from idlsave)'''
    return np.int32(struct.unpack('=i', f.read(4))[0])

def _read_int64(f):
    '''Read a signed 64-bit integer '''
    return np.int64(struct.unpack('=q', f.read(8))[0])

def _read_float32(f):
    '''Read a 32-bit float (from idlsave)'''
    return np.float32(struct.unpack('=f', f.read(4))[0])

def _read_string(f, size):
    '''Read a string of known maximum length'''
    return f.read(size).decode('utf-8').strip()

def _read_float64(f):
    '''Read a 64-bit float (from idlsave)'''
    return np.float64(struct.unpack('=d', f.read(8))[0])

def _check_val(name, got,expected):
    if got != expected:
        log.warning("{2} = {0} instead of {1}".format(got, expected, name))

def read_lmv_type2(lf):
    """ See image_def.f90 """

    header = {}
    lf.seek(12)
    # DONE before integer(kind=4) :: ijtyp(3) = 0       !  1  Image Type

    # fmt probably matters!  Default is "r4", i.e. float32 data, but could be float64
    fmt = _read_int32(lf) # 4
    # number of data blocks
    ndb = _read_int64(lf) # 5
    nhb = _read_int32(lf) # 7
    ntb = _read_int32(lf) # 8
    version_gdf = _read_int32(lf) # 9
    if version_gdf != 20:
        raise TypeError("Trying to read a version-2 file, but the version"
                        " number is {0} (should be 20)".format(version_gdf))
    type_gdf = _read_int32(lf) # 10
    dim_start = _read_int32(lf) # 11
    pad_trail = _read_int32(lf) # 12

    if dim_start % 2 == 0:
        log.warning("Got even dim_start in lmv cube: this is not expected.")
    if dim_start > 17:
        log.warning("dim_start > 17 in lmv cube: this is not expected.")

    lf.seek(16*4)
    gdf_maxdims=7
    dim_words = _read_int32(lf) # 17
    if dim_words != 2*gdf_maxdims+2:
        log.warning("dim_words = {0} instead of {1}".format(dim_words,
                                                         gdf_maxdims*2+2))
    blan_start = _read_int32(lf) # 18
    if blan_start != dim_start+dim_words+2:
        log.warning("blan_star = {0} instead of {1}".format(blan_start,
                                                         dim_start+dim_words+2))

    mdim = _read_int32(lf) # 19
    ndim = _read_int32(lf) # 20
    dims = np.fromfile(lf, count=gdf_maxdims, dtype='int64')
    if np.count_nonzero(dims) != ndim:
        raise ValueError("Disagreement between ndims and number of nonzero dims.")

    header['NAXIS'] = ndim
    valid_dims = []
    for ii,dim in enumerate(dims):
        if dim != 0:
            header['NAXIS{0}'.format(ii+1)] = dim
            valid_dims.append(ii)


    blan_words = _read_int32(lf)
    if blan_words != 2:
        log.warning("blan_words = {0} instead of 2".format(blan_words))
    extr_start = _read_int32(lf)
    bval = _read_float32(lf) # blanking value
    bval_tol = _read_float32(lf) # eval = tolerance

    # FITS requires integer BLANKs
    #header['BLANK'] = bval

    extr_words = _read_int32(lf)
    if extr_words != 6:
        log.warning("extr_words = {0} instead of 6".format(extr_words))
    coor_start = _read_int32(lf)
    if coor_start != extr_start+extr_words+2:
        log.warning("coor_start = {0} instead of {1}".format(coor_start,
                                                          extr_start+extr_words+2))
    rmin = _read_float32(lf)
    rmax = _read_float32(lf)

    # position 168
    minloc = _read_int64(lf)
    maxloc = _read_int64(lf)

    # lf.seek(184)
    coor_words = _read_int32(lf)
    if coor_words != gdf_maxdims*6:
        log.warning("coor_words = {0} instead of {1}".format(coor_words,
                                                          gdf_maxdims*6))
    desc_start = _read_int32(lf)
    if desc_start != coor_start+coor_words+2:
        log.warning("desc_start = {0} instead of {1}".format(desc_start,
                                                          coor_start+coor_words+2))

    convert = np.fromfile(lf, count=3*gdf_maxdims, dtype='float64').reshape([gdf_maxdims,3])

    # conversion of "convert" to CRPIX/CRVAL/CDELT below

    desc_words = _read_int32(lf)
    if desc_words != 3*(gdf_maxdims+1):
        log.warning("desc_words = {0} instead of {1}".format(desc_words,
                                                          3*(gdf_maxdims+1)))
    null_start = _read_int32(lf)
    if null_start != desc_start+desc_words+2:
        log.warning("null_start = {0} instead of {1}".format(null_start,
                                                          desc_start+desc_words+2))
    ijuni = _read_string(lf, 12) # data unit
    ijcode = [_read_string(lf, 12) for ii in range(gdf_maxdims)]
    pad_desc = _read_int32(lf)

    if ijuni.lower() in _bunit_dict:
        header['BUNIT'] = (_bunit_dict[ijuni.lower()],
                           ijuni)
    else:
        header['BUNIT'] = ijuni

     #! The first block length is thus
     #!	s_dim-1 + (2*mdim+4) + (4) + (8) +  (6*mdim+2) + (3*mdim+5)
     #! = s_dim-1 + mdim*(2+6+3) + (4+4+2+5+8)
     #! = s_dim-1 + 11*mdim + 23
     #! With mdim = 7, s_dim=11, this is 110 spaces
     #! With mdim = 8, s_dim=11, this is 121 spaces
     #! MDIM > 8 would NOT fit in one block...
     #!
     #! Block 2: Ancillary information
     #!
     #! The same logic of Length + Pointer is used there too, although the
     #! length are fixed. Note rounding to even number for the pointer offsets
     #! in order to preserve alignement...
     #!

    lf.seek(512)
    posi_words = _read_int32(lf)
    _check_val('posi_words', posi_words, 15)

    proj_start = _read_int32(lf)
    source_name = _read_string(lf, 12)
    header['OBJECT'] = source_name
    coordinate_system = _read_string(lf, 12)

    header['RA'] = _read_float64(lf)
    header['DEC'] = _read_float64(lf)
    header['LII'] = _read_float64(lf)
    header['BII'] = _read_float64(lf)
    header['EPOCH'] = _read_float32(lf)
    #pad_posi = _read_float32(lf)
    #print pad_posi
    #raise ValueError("pad_posi should probably be 0?")

    #! PROJECTION
    #integer(kind=4) :: proj_words = 9     ! Projection length: 9 used + 1 padding
    #integer(kind=4) :: spec_start !! = proj_start + 12
    #real(kind=8) :: a0      = 0.d0        ! 89  X of projection center
    #real(kind=8) :: d0      = 0.d0        ! 91  Y of projection center
    #real(kind=8) :: pang    = 0.d0        ! 93  Projection angle
    #integer(kind=4) :: ptyp = p_none      ! 88  Projection type (see p_... codes)
    #integer(kind=4) :: xaxi = 0           ! 95  X axis
    #integer(kind=4) :: yaxi = 0           ! 96  Y axis
    #integer(kind=4) :: pad_proj
    #!

    proj_words = _read_int32(lf)
    spec_start = _read_int32(lf)
    _check_val('spec_start', spec_start, proj_start+proj_words+2)
    if proj_words == 9:
        header['PROJ_A0'] = _read_float64(lf)
        header['PROJ_D0'] = _read_float64(lf)
        header['PROJPANG'] = _read_float64(lf)
        ptyp = _read_int32(lf)
        header['PROJXAXI'] = _read_int32(lf)
        header['PROJYAXI'] = _read_int32(lf)
    elif proj_words != 0:
        raise ValueError("Invalid # of projection keywords")

    for kw in header:
        if 'CTYPE' in kw:
            if header[kw].strip() in cel_types:
                n_dashes = 5-len(header[kw].strip())
                header[kw] = header[kw].strip()+ '-'*n_dashes + _proj_dict[ptyp]

    for ii,((ref,val,inc),code) in enumerate(zip(convert,ijcode)):
        if ii in valid_dims:
            # jul14a gio/to_imfits.f90 line 284-313
            if ptyp != 0 and (ii+1) in (header['PROJXAXI'],
                                        header['PROJYAXI']):
                #! Compute reference pixel so that VAL(REF) = 0
                ref = ref - val/inc
                if (ii+1) == header['PROJXAXI']:
                    val = header['PROJ_A0']
                elif (ii+1) == header['PROJYAXI']:
                    val = header['PROJ_D0']
                else:
                    raise ValueError("Impossible state - code bug.")
                val = val*r2deg
                inc = inc*r2deg
                rota = r2deg*header['PROJPANG']
            elif code in ('RA', 'L', 'B', 'DEC', 'LII', 'BII', 'GLAT',
                          'GLON', 'LAT', 'LON'):
                val = val*r2deg
                inc = inc*r2deg
                rota = 0.0
            # These are not implemented: prefer to maintain original units (we're
            # reading in to spectral_cube after all, no need to change units until the
            # output step)
            #elseif (code.eq.'FREQUENCY') then
            #val = val*1.0d6            ! MHz to Hz
            #inc = inc*1.0d6
            #elseif (code.eq.'VELOCITY') then
            #code = 'VRAD'              ! force VRAD instead of VELOCITY for CASA
            #val = val*1.0d3            ! km/s to m/s
            #inc = inc*1.0d3

            header['CRPIX{0}'.format(ii+1)] = ref
            header['CRVAL{0}'.format(ii+1)] = val
            header['CDELT{0}'.format(ii+1)] = inc


    for ii,ctype in enumerate(ijcode):
        if ii in valid_dims:
            header['CTYPE{0}'.format(ii+1)] = _ctype_dict[ctype]
            header['CUNIT{0}'.format(ii+1)] = _cunit_dict[ctype]

    spec_words = _read_int32(lf)
    reso_start = _read_int32(lf)
    _check_val('reso_start', reso_start, proj_start+proj_words+2+spec_words+2)
    if spec_words == 14:
        header['FRES'] = _read_float64(lf)
        header['FIMA'] = _read_float64(lf)
        header['FREQ'] = _read_float64(lf)
        header['VRES'] = _read_float32(lf)
        header['VOFF'] = _read_float32(lf)
        header['DOPP'] = _read_float32(lf)
        header['FAXI'] = _read_int32(lf)
        header['LINENAME'] = _read_string(lf, 12)
        header['VTYPE'] = _read_int32(lf)
    elif spec_words != 0:
        raise ValueError("Invalid # of spectroscopic keywords")

     #! SPECTROSCOPY
     #integer(kind=4) :: spec_words  = 14   ! Spectroscopy length: 14 used
     #integer(kind=4) :: reso_start  !! = spec_words + 16
     #real(kind=8) :: fres       = 0.d0     !101  Frequency resolution
     #real(kind=8) :: fima       = 0.d0     !103  Image frequency
     #real(kind=8) :: freq       = 0.d0     !105  Rest Frequency
     #real(kind=4) :: vres       = 0.0      !107  Velocity resolution
     #real(kind=4) :: voff       = 0.0      !108  Velocity offset
     #real(kind=4) :: dopp       = 0.0      !     Doppler factor
     #integer(kind=4) :: faxi    = 0        !109  Frequency axis
     #integer(kind=4) :: ijlin(3) = 0       ! 98  Line name
     #integer(kind=4) :: vtyp    = vel_unk  ! Velocity type (see vel_... codes)

    reso_words = _read_int32(lf)
    nois_start = _read_int32(lf)
    _check_val('nois_start', nois_start, proj_start+proj_words+2+spec_words+2+reso_words+2)
    if reso_words == 3:
        header['BMAJ'] = _read_float32(lf)
        header['BMIN'] = _read_float32(lf)
        header['BPA'] = _read_float32(lf)
        #pad_reso = _read_float32(lf)
    elif reso_words != 0:
        raise ValueError("Invalid # of resolution keywords")

     #! RESOLUTION
     #integer(kind=4) :: reso_words = 3     ! Resolution length: 3 used + 1 padding
     #integer(kind=4) :: nois_start !! = reso_words + 6
     #real(kind=4) :: majo    = 0.0         !111  Major axis
     #real(kind=4) :: mino    = 0.0         !112  Minor axis
     #real(kind=4) :: posa    = 0.0         !113  Position angle
     #real(kind=4) :: pad_reso


    nois_words = _read_int32(lf)
    astr_start = _read_int32(lf)
    _check_val('astr_start', astr_start, proj_start+proj_words+2+spec_words+2+reso_words+2+nois_words+2)
    if nois_words == 2:
        header['NOISE_T'] = (_read_float32(lf), "Theoretical Noise")
        header['NOISERMS'] = (_read_float32(lf), "Measured (RMS) noise")
    elif nois_words != 0:
        raise ValueError("Invalid # of noise keywords")

     #! NOISE
     #integer(kind=4) :: nois_words = 2     ! Noise section length: 2 used
     #integer(kind=4) :: astr_start !! = s_nois + 4
     #real(kind=4) :: noise   = 0.0         ! 115 Theoretical noise
     #real(kind=4) :: rms     = 0.0         ! 116 Actual noise

    astr_words = _read_int32(lf)
    uvda_start = _read_int32(lf)
    _check_val('uvda_start', uvda_start, proj_start+proj_words+2+spec_words+2+reso_words+2+nois_words+2+astr_words+2)
    if astr_words == 3:
        header['MURA'] = _read_float32(lf)
        header['MUDEC'] = _read_float32(lf)
        header['PARALLAX'] = _read_float32(lf)
    elif astr_words != 0:
        raise ValueError("Invalid # of astrometry keywords")

     #! ASTROMETRY
     #integer(kind=4) :: astr_words = 3    ! Proper motion section length: 3 used + 1 padding
     #integer(kind=4) :: uvda_start !! = s_astr + 4
     #real(kind=4) :: mura     = 0.0        ! 118 along RA, in mas/yr
     #real(kind=4) :: mudec    = 0.0        ! 119 along Dec, in mas/yr
     #real(kind=4) :: parallax = 0.0        ! 120 in mas
     #real(kind=4) :: pad_astr
     #! real(kind=4) :: pepoch   = 2000.0     ! 121 in yrs ?

    code_uvt_last=25
    uvda_words = _read_int32(lf)
    void_start = _read_int32(lf)
    _check_val('void_start', void_start, proj_start + proj_words + 2 +
               spec_words + 2 + reso_words + 2 + nois_words + 2 + astr_words +
               2 + uvda_words + 2)
    if uvda_words == 18+2*code_uvt_last:
        version_uv = _read_int32(lf)
        nchan = _read_int32(lf)
        nvisi = _read_int64(lf)
        nstokes = _read_int32(lf)
        natom = _read_int32(lf)
        basemin = _read_float32(lf)
        basemax = _read_float32(lf)
        fcol = _read_int32(lf)
        lcol = _read_int32(lf)
        nlead = _read_int32(lf)
        ntrail = _read_int32(lf)
        column_pointer = np.fromfile(lf, count=code_uvt_last, dtype='int32')
        column_size = np.fromfile(lf, count=code_uvt_last, dtype='int32')
        column_codes = np.fromfile(lf, count=nlead+ntrail, dtype='int32')
        column_types = np.fromfile(lf, count=nlead+ntrail, dtype='int32')
        order = _read_int32(lf)
        nfreq = _read_int32(lf)
        atoms = np.fromfile(lf, count=4, dtype='int32')

    elif uvda_words != 0:
        raise ValueError("Invalid # of UV data keywords")

     #! UV_DATA information
     #integer(kind=4) :: uvda_words  = 18+2*code_uvt_last ! Length of section: 14 used
     #integer(kind=4) :: void_start  !! = s_uvda + l_uvda + 2
     #integer(kind=4) :: version_uv = code_version_uvt_current  ! 1 version number. Will allow us to change the data format
     #integer(kind=4) :: nchan = 0         ! 2 Number of channels
     #integer(kind=8) :: nvisi = 0         ! 3-4 Independent of the transposition status
     #integer(kind=4) :: nstokes = 0       ! 5 Number of polarizations
     #integer(kind=4) :: natom = 0         ! 6. 3 for real, imaginary, weight. 1 for real.
     #real(kind=4)    :: basemin = 0.      ! 7 Minimum Baseline
     #real(kind=4)    :: basemax = 0.      ! 8 Maximum Baseline
     #integer(kind=4) :: fcol              ! 9 Column of first channel
     #integer(kind=4) :: lcol              ! 10 Column of last  channel
     #! The number of information per channel can be obtained by
     #!       (lcol-fcol+1)/(nchan*natom)
     #! so this could allow to derive the number of Stokes parameters
     #! Leading data at start of each visibility contains specific information
     #integer(kind=4) :: nlead = 7           ! 11 Number of leading informations (at lest 7)
     #! Trailing data at end of each visibility may hold additional information
     #integer(kind=4) :: ntrail = 0          ! 12 Number of trailing informations
     #!
     #! Leading / Trailing information codes have been specified before
     #integer(kind=4) :: column_pointer(code_uvt_last) = code_null ! Back pointer to the columns...
     #integer(kind=4) :: column_size(code_uvt_last) = 0  ! Number of columns for each
     #! In the data, we instead have the codes for each column
     #! integer(kind=4) :: column_codes(nlead+ntrail)         ! Start column for each ...
     #! integer(kind=4) :: column_types(nlead+ntrail) /0,1,2/ ! Number of columns for each: 1 real*4, 2 real*8
     #! Leading / Trailing information codes
     #!
     #integer(kind=4) :: order = 0          ! 13  Stoke/Channel ordering
     #integer(kind=4) :: nfreq = 0          ! 14  ! 0 or = nchan*nstokes
     #integer(kind=4) :: atoms(4)           ! 15-18 Atom description
     #!
     #real(kind=8), pointer :: freqs(:) => null()     ! (nchan*nstokes) = 0d0
     #integer(kind=4), pointer :: stokes(:) => null() ! (nchan*nstokes) or (nstokes) = code_stoke
     #!
     #real(kind=8), pointer :: ref(:) => null()
     #real(kind=8), pointer :: val(:) => null()
     #real(kind=8), pointer :: inc(:) => null()

    lf.seek(1024)
    real_dims = dims[:ndim]
    data = np.fromfile(lf, count=np.prod(real_dims),
                       dtype='float32').reshape(real_dims[::-1])
    data[data==bval] = np.nan

    return data,header


io_registry.register_reader('lmv', BaseSpectralCube, load_lmv_cube)
io_registry.register_reader('class_lmv', BaseSpectralCube, load_lmv_cube)
io_registry.register_identifier('lmv', BaseSpectralCube, is_lmv)
