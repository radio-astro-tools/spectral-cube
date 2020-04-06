import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time

__all__ = ['wcs_casa2astropy']

EQUATORIAL_SYSTEMS = ['B1950', 'B1950_VLA', 'J2000', 'ICRS']

SPECSYS = {}
SPECSYS['BARY'] = 'BARYCENT'
SPECSYS['TOPO'] = 'TOPOCENT'
SPECSYS['LSRK'] = 'LSRK'
SPECSYS['LSRD'] = 'LSRD'
SPECSYS['GEO'] = 'GEOCENTR'
SPECSYS['GALACTO'] = 'GALACTOC'
SPECSYS['LGROUP'] = 'LOCALGRP'
SPECSYS['CMB'] = 'CMBDIPOL'
SPECSYS['REST'] = 'SOURCE'
SPECSYS['Unknown'] = 'UKN'

RADESYS = {}
RADESYS['J2000'] = 'FK5'
RADESYS['B1950'] = 'FK4'
RADESYS['B1950_VLA'] = 'FK4'
RADESYS['ICRS'] = 'ICRS'

EQUINOX = {}
EQUINOX['J2000'] = 2000.
EQUINOX['B1950'] = 1950.
EQUINOX['B1950_VLA'] = 1979.9

AXES_TO_CTYPE = {
    'Frequency': 'FREQ',
    'GALACTIC': {'Longitude': 'GLON',
                 'Latitude': 'GLAT'},
    'SUPERGAL': {'Longitude': 'SLON',
                 'Latitude': 'SLAT'},
    'ECLIPTIC': {'Longitude': 'ELON',
                 'Latitude': 'ELAT'}
}

for system in EQUATORIAL_SYSTEMS:
    AXES_TO_CTYPE[system] = {'Right Ascension': 'RA--',
                             'Declination': 'DEC-'}


def sanitize_unit(unit):
    if unit == "'":
        return 'arcmin'
    elif unit == '"':
        return 'arcsec'
    else:
        return unit


def wcs_casa2astropy(coordsys):
    """
    Convert a casac.coordsys object into an astropy.wcs.WCS object
    """

    # Rather than try and parse the CASA coordsys ourselves, we delegate
    # to CASA by getting it to write out a FITS file and reading back in
    # using WCS

    header = fits.Header()

    # Observer information (ObsInfo.cc)

    header['OBSERVER'] = coordsys['observer']
    header['TELESCOP'] = coordsys['telescope']

    if coordsys['obsdate']['refer'] == 'LAST':

        header['TIMESYS'] = 'UTC'

    else:

        header['TIMESYS'] = coordsys['obsdate']['refer']
        header['MJD-OBS'] = coordsys['obsdate']['m0']['value']

        dateobs = Time(header['MJD-OBS'],
                       format='mjd',
                       scale=coordsys['obsdate']['refer'].lower())
        dateobs.precision = 6
        header['DATE-OBS'] = dateobs.isot

    if 'telescopeposition' in coordsys:

        obsgeo_lon = coordsys['telescopeposition']['m0']['value']
        obsgeo_lat = coordsys['telescopeposition']['m1']['value']
        obsgeo_alt = coordsys['telescopeposition']['m2']['value']

        header['OBSGEO-X'] = obsgeo_alt * np.cos(obsgeo_lon) * np.cos(obsgeo_lat)
        header['OBSGEO-Y'] = obsgeo_alt * np.sin(obsgeo_lon) * np.cos(obsgeo_lat)
        header['OBSGEO-Z'] = obsgeo_alt * np.sin(obsgeo_lat)

    # World coordinates

    # Find worldmap entries

    worldmap = {}

    for key, value in coordsys.items():
        if key.startswith('worldmap'):
            index = int(key[8:])
            worldmap[index] = value

    # Now iterate through the different coordinate types to populate the WCS

    header['WCSAXES'] = np.max([np.max(idx) + 1 for idx in worldmap.values()])

    # Initialize PC
    for ii in range(header['WCSAXES']):
        for jj in range(header['WCSAXES']):
            header[f'PC{ii+1}_{jj+1}'] = 0.

    for coord_type in ('direction', 'spectral', 'stokes', 'linear'):

        # Each coordinate type is stored with a numerical index,
        # so for example a cube might have direction0 and
        # spectral1, or spectral0, direction1, stokes2. The actual
        # index is irrelevant for the rest of the loop.
        for index in range(len(worldmap)):
            if f'{coord_type}{index}' in coordsys:
                data = coordsys[f'{coord_type}{index}']
                break
        else:
            # Skip the rest of the loop for the current coord_type
            # since the coord_type wasn't found with any index.
            continue

        if coord_type == 'direction':
            idx1, idx2 = worldmap[index] + 1
            header[f'CTYPE{idx1}'] = AXES_TO_CTYPE[data['system']][data['axes'][0]] + '-' + data['projection']
            header[f'CTYPE{idx2}'] = AXES_TO_CTYPE[data['system']][data['axes'][1]] + '-' + data['projection']
            header[f'CRPIX{idx1}'] = data['crpix'][0] + 1
            header[f'CRPIX{idx2}'] = data['crpix'][1] + 1
            header[f'CRVAL{idx1}'] = data['crval'][0]
            header[f'CRVAL{idx2}'] = data['crval'][1]
            header[f'CDELT{idx1}'] = data['cdelt'][0]
            header[f'CDELT{idx2}'] = data['cdelt'][1]
            header[f'CUNIT{idx1}'] = sanitize_unit(data['units'][0])
            header[f'CUNIT{idx2}'] = sanitize_unit(data['units'][1])
            header['LONPOLE'] = data['longpole']
            header['LATPOLE'] = data['latpole']
            if data.get('system') in EQUATORIAL_SYSTEMS:
                header['RADESYS'] = RADESYS[data['conversionSystem']]
                if data['conversionSystem'] in EQUINOX:
                    header['EQUINOX'] = EQUINOX[data['conversionSystem']]
            # NOTE: unclear if it is deliberate that the following is always
            # ?_2 and ?_1 or whether it should depend on the index of the
            # longitude.
            if data.get('system') in EQUATORIAL_SYSTEMS:
                header[f'PV{idx2}_1'] = 0.
                header[f'PV{idx2}_2'] = 0.
            header[f'PC{idx1}_{idx1}'] = data['pc'][0, 0]
            header[f'PC{idx1}_{idx2}'] = data['pc'][0, 1]
            header[f'PC{idx2}_{idx1}'] = data['pc'][1, 0]
            header[f'PC{idx2}_{idx2}'] = data['pc'][1, 1]
        elif coord_type == 'stokes':
            idx = worldmap[index][0] + 1
            header[f'CTYPE{idx}'] = 'STOKES'
            header[f'CRVAL{idx}'] = data['crval'][0]
            header[f'CRPIX{idx}'] = data['crpix'][0] + 1
            header[f'CDELT{idx}'] = data['cdelt'][0]
            header[f'CUNIT{idx}'] = ''
            header[f'PC{idx}_{idx}'] = data['pc'][0][0]
        elif coord_type == 'spectral':
            idx = worldmap[index][0] + 1
            if 'tabular' in data:
                header[f'CTYPE{idx}'] = AXES_TO_CTYPE[data['tabular']['axes'][0]]
                header[f'CRVAL{idx}'] = data['tabular']['crval'][0]
                header[f'CRPIX{idx}'] = data['tabular']['crpix'][0] + 1
                # It looks like we can't just use:
                # header[f'CDELT{idx}'] = data['tabular']['cdelt'][0]
                # because this doesn't match what appears in a FITS header
                # exported from CASA. Instead we use the interval between the
                # first two tabulated values. See
                # https://github.com/radio-astro-tools/spectral-cube/issues/614
                # for more information.
                header[f'CDELT{idx}'] = np.diff(data['tabular']['worldvalues'][:2])[0]
                header[f'CUNIT{idx}'] = data['tabular']['units'][0]
            else:
                header[f'CTYPE{idx}'] = data['wcs']['ctype']
                header[f'CRVAL{idx}'] = data['wcs']['crval']
                header[f'CRPIX{idx}'] = data['wcs']['crpix'] + 1
                header[f'CDELT{idx}'] = data['wcs']['cdelt']
                header[f'CUNIT{idx}'] = data['unit']
            header[f'PC{idx}_{idx}'] = 1.0
            header[f'RESTFRQ'] = data['restfreq']
            header[f'SPECSYS'] = SPECSYS[data['system']]
        elif coord_type == 'linear':
            indices = worldmap[index] + 1
            for idx1 in range(len(indices)):
                header[f'CTYPE{indices[idx1]}'] = data['axes'][idx1].upper()
                header[f'CRVAL{indices[idx1]}'] = data['crval'][idx1]
                header[f'CRPIX{indices[idx1]}'] = data['crpix'][idx1] + 1
                header[f'CDELT{indices[idx1]}'] = data['cdelt'][idx1]
                header[f'CUNIT{indices[idx1]}'] = data['units'][idx1]
                for idx2 in range(len(indices)):
                    header[f'PC{indices[idx1]}_{indices[idx1]}'] = data['pc'][idx1, idx1]
                    header[f'PC{indices[idx1]}_{indices[idx2]}'] = data['pc'][idx1, idx2]
                    header[f'PC{indices[idx2]}_{indices[idx1]}'] = data['pc'][idx2, idx1]
                    header[f'PC{indices[idx2]}_{indices[idx2]}'] = data['pc'][idx2, idx2]
        else:
            raise NotImplementedError(f'coord_type is {coord_type}')

    return WCS(header)
