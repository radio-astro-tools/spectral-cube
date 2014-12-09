import numpy as np
from astropy.io import fits
import tempfile
import warnings

from ..wcs_utils import add_stokes_axis_to_wcs


def make_casa_mask(SpecCube, outname, append_to_image=True,
                   img=None, add_stokes=True):
    '''
    Takes a SpectralCube object as an input. Outputs the mask in a CASA
    friendly form.

    Parameters
    ----------
    SpecCube : SpectralCube
        SpectralCube object containing mask.
    outname : str
        Name of the outputted mask file.
    append_to_image : bool, optional
        Appends the mask to a given image.
    img : str, optional
        Image to be appended to. Must be specified if append_to_image is
        enabled.
    add_stokes: bool, optional
        Adds a Stokes axis onto the wcs from SpecCube.
    '''

    try:
        from taskinit import ia
        from tasks import immath
    except ImportError:
        print("Cannot import casac. Must be run in a CASA environment.")

    # Get the header info from the image
    # There's not wcs_astropy2casa (yet), so create a temporary file for
    # CASA to open.
    temp = tempfile.NamedTemporaryFile()
    # CASA is closing this file at some point so set it to manual delete.
    temp2 = tempfile.NamedTemporaryFile(delete=False)

    # Grab wcs
    # Optionally re-add on the Stokes axis
    if add_stokes:
        wcs = SpecCube.wcs
        new_wcs = add_stokes_axis_to_wcs(wcs, wcs.wcs.naxis)
        header = new_wcs.to_header()
        shape = (1,) + SpecCube.shape
    else:
        # Just grab the header from SpecCube
        header = SpecCube.header
        shape = SpecCube.shape

    hdu = fits.PrimaryHDU(header=header,
                          data=np.empty(shape))

    hdu.writeto(temp.name)

    ia.fromfits(infile=temp.name, outfile=temp2.name, overwrite=True)

    temp.close()

    cs = ia.coordsys()

    ia.close()

    temp2.close()

    mask_arr = SpecCube.mask.include()

    # Reshape mask with possible Stokes axis
    mask_arr = mask_arr.reshape(shape)

    # Transpose to match CASA axes
    mask_arr = mask_arr.T

    # CASA doesn't like bool? Using floats for now...
    ia.newimagefromarray(outfile=outname,
                         pixels=mask_arr.astype('float64'))

    ia.open(outname)
    ia.setcoordsys(cs.torecord())

    ia.close()

    if append_to_image:
        if img is None:
            raise TypeError("img argument must be specified to append the mask.")
        warnings.warn("Image appending not working yet.")
        # immath(imagename=img, mode='evalexpr', expr='IM0',
        #        outfile=img+"_test", mask='mask('+outname+')')
