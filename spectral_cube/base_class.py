from astropy import units as u

from . import wcs_utils

DOPPLER_CONVENTIONS = {}
DOPPLER_CONVENTIONS['radio'] = u.doppler_radio
DOPPLER_CONVENTIONS['optical'] = u.doppler_optical
DOPPLER_CONVENTIONS['relativistic'] = u.doppler_relativistic



class BaseNDClass(object):
    @property
    def _nowcs_header(self):
        """
        Return a copy of the header with no WCS information attached
        """
        return wcs_utils.strip_wcs_from_header(self._header)
