import warnings
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from spectral_cube import SpectralCube,SpectralCubeMask

def load_fits_cube(filename, extnum=0, **kwargs):
    """
    Read in a cube from a FITS file using astropy.
    
    Parameters
    ----------
    filename: str
        The FITS cube file name
    extnum: int
        The extension number containing the data to be read
    kwargs: dict
        passed to fits.open
    """

    # open the file
    hdulist = fits.open(filename, **kwargs)

    # read the data - assume first extension
    data = hdulist[extnum].data

    # note where data is valid
    valid = np.isfinite(data)

    # note the header and WCS information
    hdr = hdulist[extnum].header
    wcs = WCS(hdr)

    metadata = {'filename':filename, 'extension_number':extnum}

    mask = SpectralCubeMask(wcs, np.logical_not(valid))
    cube = SpectralCube(data, wcs, mask, metadata=metadata)
    return cube
    
    
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# CASA input/output
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Read and write from a CASA image. This has a few
# complications. First, by default CASA does not return the
# "python order" and so we either have to transpose the cube on
# read or have dueling conventions. Second, CASA often has
# degenerate stokes axes present in unpredictable places (3rd or
# 4th without a clear expectation). We need to replicate these
# when writing but don't want them in memory. By default, try to
# yield the same array in memory that we would get from astropy.

def wcs_casa2astropy(casa_wcs):
    """
    Convert a casac.coordsys object into an astropy.wcs.WCS object
    """

    from astropy.wcs import WCS

    wcs = WCS(naxis=int(casa_wcs.naxes()))

    crpix = casa_wcs.referencepixel()
    if crpix['ar_type'] != 'absolute':
        raise ValueError("Unexpected ar_type: %s" % crpix['ar_type'])
    elif crpix['pw_type'] != 'pixel':
        raise ValueError("Unexpected pw_type: %s" % crpix['pw_type'])
    else:
        wcs.wcs.crpix = crpix['numeric']

    cdelt = casa_wcs.increment()
    if cdelt['ar_type'] != 'absolute':
        raise ValueError("Unexpected ar_type: %s" % cdelt['ar_type'])
    elif cdelt['pw_type'] != 'world':
        raise ValueError("Unexpected pw_type: %s" % cdelt['pw_type'])
    else:
        wcs.wcs.cdelt = cdelt['numeric']

    crval = casa_wcs.referencevalue()
    if crval['ar_type'] != 'absolute':
        raise ValueError("Unexpected ar_type: %s" % crval['ar_type'])
    elif crval['pw_type'] != 'world':
        raise ValueError("Unexpected pw_type: %s" % crval['pw_type'])
    else:
        wcs.wcs.crval = crval['numeric']

    wcs.wcs.cunit = casa_wcs.units()

    # mapping betweeen CASA and FITS
    COORD_TYPE = {}
    COORD_TYPE['Right Ascension'] = "RA--"
    COORD_TYPE['Declination'] = "DEC-"
    COORD_TYPE['Frequency'] = "FREQ"
    COORD_TYPE['Stokes'] = "STOKES"

    # There is no easy way at the moment to extract the orginal projection
    # codes from a coordsys object, so we need to figure out how to do this in
    # the most general way. The code below is still experimental.
    ctype = []
    for i, name in enumerate(casa_wcs.names()):
        if name in COORD_TYPE:
            ctype.append(COORD_TYPE[name])
            if casa_wcs.axiscoordinatetypes()[i] == 'Direction':
                ctype[-1] += ("%4s" % casa_wcs.projection()['type']).replace(' ', '-')
        else:
            raise KeyError("Don't know how to convert: %s" % name)

    wcs.wcs.ctype = ctype

    return wcs

try:
    #import casac
    from tasks import ia
    #from task_init import ia

    def from_casa_image(filename, dropdeg=True, skipdata=False,
                        skipvalid=False, skipcs=False):
        """
        Load a cube (into memory?) from a CASA image. By default it will transpose
        the cube into a 'python' order and drop degenerate axes. These options can
        be suppressed. The object holds the coordsys object from the image in
        memory.
        """

        # use the ia tool to get the file contents
        ia.open(filename)

        # read in the data
        if not skipdata:
            data = ia.getchunk(dropdeg=dropdeg)

        # CASA stores validity of data as a mask
        if not skipvalid:
            valid = ia.getchunk(getmask=True, dropdeg=dropdeg)

        # transpose is dealt with within the cube object
            
        # read in coordinate system object
        casa_cs = ia.coordsys()

        wcs = wcs_casa2astropy(casa_cs)

        # don't need this yet
        # stokes = get_casa_axis(temp_cs, wanttype="Stokes", skipdeg=False,)

        #    if stokes == None:
        #        order = np.arange(self.data.ndim)
        #    else:
        #        order = []
        #        for ax in np.arange(self.data.ndim+1):
        #            if ax == stokes:
        #                continue
        #            order.append(ax)

        #    self.casa_cs = ia.coordsys(order)
            
            # This should work, but coordsys.reorder() has a bug
            # on the error checking. JIRA filed. Until then the
            # axes will be reversed from the original.

            #if transpose == True:
            #    new_order = np.arange(self.data.ndim)
            #    new_order = new_order[-1*np.arange(self.data.ndim)-1]
            #    print new_order
            #    self.casa_cs.reorder(new_order)
        
        # close the ia tool
        ia.close()

        metadata = {'filename':filename}

        mask = SpectralCubeMask(wcs, np.logical_not(valid))
        cube = SpectralCube(data, wcs, mask, metadata=metadata)

        return cube
except ImportError:
    warnings.warn("Could not import CASA (casac) and therefore cannot read CASA .image files")
