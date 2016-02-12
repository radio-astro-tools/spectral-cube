
class BaseNDClass(object):
    @property
    def _nowcs_header(self):
        """
        Return a copy of the header with no WCS information attached
        """
        return wcs_utils.strip_wcs_from_header(self._header)

