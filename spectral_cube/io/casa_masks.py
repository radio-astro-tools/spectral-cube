import numpy as np
from astropy.io import fits
import tempfile


def make_casa_mask(SpecCube, outname, append_to_image=True,
                   img=""):
    '''
    Takes a SpectralCube object as an input. Outputs the mask in a CASA
    friendly form.
    '''

    try:
        from taskinit import ia
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
        ia.open(img)
        ia.maskhandler('set', outname)
        ia.done()
