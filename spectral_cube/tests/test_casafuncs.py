import pytest
import numpy as np
from numpy.testing import assert_allclose
import os

from ..io.casa_masks import make_casa_mask
from ..io.casa_image import wcs_casa2astropy
from .. import SpectralCube, BooleanArrayMask

try:
    from taskinit import ia
    casaOK = True
except ImportError:
    print("Run in CASA environment.")
    casaOK = False


def make_casa_testimage(infile, outname):

    ia.fromfits(infile=infile, outfile=outname, overwrite=True)
    ia.close()


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
def test_casa_mask():

    cube = SpectralCube.read('adv.fits')

    mask_array = np.array([[True, False], [False, False], [True, True]])
    bool_mask = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                 shape=cube.shape)
    cube = cube.with_mask(bool_mask)

    if os.path.exists('casa.mask'):
        os.system('rm -rf casa.mask')

    make_casa_mask(cube, 'casa.mask', add_stokes=False,
                   append_to_image=False)

    ia.open('casa.mask')

    casa_mask = ia.getchunk()

    coords = ia.coordsys()

    ia.close()

    # Test masks
    # Mask array is broadcasted to the cube shape. Mimic this, switch to ints,
    # and transpose to match CASA image.
    compare_mask = np.tile(mask_array, (4, 1, 1)).astype('int16').T
    assert np.all(compare_mask == casa_mask)

    # Test WCS info

    # Convert back to an astropy wcs object so transforms are dealt with.
    casa_wcs = wcs_casa2astropy(coords)
    header = casa_wcs.to_header()  # Invokes transform

    # Compare some basic properties EXCLUDING the spectral axis
    assert np.allclose(cube.wcs.wcs.crval[:2], casa_wcs.wcs.crval[:2])
    assert np.all(cube.wcs.wcs.cdelt[:2] == casa_wcs.wcs.cdelt[:2])
    assert np.all(list(cube.wcs.wcs.cunit)[:2] == list(casa_wcs.wcs.cunit)[:2])
    assert np.all(list(cube.wcs.wcs.ctype)[:2] == list(casa_wcs.wcs.ctype)[:2])

    # Reference pixels in CASA are 1 pixel off.
    assert_allclose(cube.wcs.wcs.crpix, casa_wcs.wcs.crpix,
                    atol=1.0)


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
def test_casa_mask_append():

    cube = SpectralCube.read('adv.fits')

    mask_array = np.array([[True, False], [False, False], [True, True]])
    bool_mask = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                 shape=cube.shape)
    cube = cube.with_mask(bool_mask)

    make_casa_testimage('adv.fits', 'casa.image')

    if os.path.exists('casa.mask'):
        os.system('rm -rf casa.mask')

    make_casa_mask(cube, 'casa.mask', append_to_image=True,
                   img='casa.image', add_stokes=False)

    assert os.path.exists('casa.image/casa.mask')
