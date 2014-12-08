import numpy as np
from astropy.io import fits
import tempfile
import warnings


def make_casa_mask(SpecCube, outname, append_to_image=True,
                   img=""):
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
    '''

    try:
        from taskinit import ia
        from tasks import immath
    except ImportError:
        print("Run in CASA! Cannot import casac.")

    # Get the header info from the image
    # There's not wcs_astropy2casa (yet), so create a temporary file for
    # CASA to open.
    temp = tempfile.NamedTemporaryFile()
    # CASA is closing this file at some point so set it to manual delete.
    temp2 = tempfile.NamedTemporaryFile(delete=False)

    hdu = fits.PrimaryHDU(header=SpecCube.header,
                          data=np.empty(SpecCube.shape))

    hdu.writeto(temp.name)

    ia.fromfits(infile=temp.name, outfile=temp2.name, overwrite=True)

    temp.close()

    cs = ia.coordsys()

    ia.close()

    temp2.close()

    mask_arr = SpecCube.mask.include()

    # CASA doesn't like bool? Using floats for now...
    ia.newimagefromarray(outfile=outname,
                         pixels=mask_arr.astype('float64'))

    ia.open(outname)
    ia.setcoordsys(cs.torecord())

    ia.close()

    if append_to_image:
        warnings.warn("Image appending not working yet.")
        # immath(imagename=img, mode='evalexpr', expr='IM0',
        #        outfile=img+"_test", mask='mask('+outname+')')
