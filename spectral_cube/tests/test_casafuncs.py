from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
from numpy.testing import assert_allclose
import os

from astropy import units as u

from ..io.casa_masks import make_casa_mask
from ..io.casa_image import wcs_casa2astropy
from .. import SpectralCube, BooleanArrayMask, VaryingResolutionSpectralCube
from . import path

try:
    import casatools
    ia = casatools.image()
    casaOK = True
except ImportError:
    try:
        from taskinit import ia
        casaOK = True
    except ImportError:
        print("Run in CASA environment.")
        casaOK = False


def make_casa_testimage(infile, outname):

    if not casaOK:
        raise Exception("Attempted to make a CASA test image in a non-CASA "
                        "environment")
    ia.fromfits(infile=infile, outfile=outname, overwrite=True)
    ia.close()

    cube = SpectralCube.read(infile)
    if isinstance(cube, VaryingResolutionSpectralCube):
        ia.open(outname)
        # populate restoring beam emptily
        ia.setrestoringbeam(major={'value':1.0, 'unit':'arcsec'},
                            minor={'value':1.0, 'unit':'arcsec'},
                            pa={'value':90.0, 'unit':'deg'},
                            channel=len(cube.beams)-1,
                            polarization=-1,
                           )
        # populate each beam (hard assumption of 1 poln)
        for channum, beam in enumerate(cube.beams):
            casabdict = {'major': {'value':beam.major.to(u.deg).value, 'unit':'deg'},
                         'minor': {'value':beam.minor.to(u.deg).value, 'unit':'deg'},
                         'positionangle': {'value':beam.pa.to(u.deg).value, 'unit':'deg'}
                        }
            ia.setrestoringbeam(beam=casabdict, channel=channum, polarization=0)

        ia.close()


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
@pytest.mark.parametrize('basefn', ('adv', 'advs', 'sdav', 'vad', 'vsad'))
def test_casa_read(basefn):

    cube = SpectralCube.read(path('{0}.fits').format(basefn))

    make_casa_testimage(path('{0}.fits').format(basefn), path('casa_{0}.image').format(basefn))

    casacube = SpectralCube.read(path('casa_{0}.image').format(basefn), format='casa_image')

    assert casacube.shape == cube.shape
    # what other equalities should we check?

    os.system('rm -rf {0}'.format(path('casa_{0}.image'.format(basefn))))

@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
def test_casa_mask():

    cube = SpectralCube.read(path('adv.fits'))

    mask_array = np.array([[True, False], [False, False], [True, True]])
    bool_mask = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                 shape=cube.shape)
    cube = cube.with_mask(bool_mask)

    if os.path.exists('casa.mask'):
        os.system('rm -rf casa.mask')

    make_casa_mask(cube, 'casa.mask', add_stokes=False,
                   append_to_image=False, overwrite=True)

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
    casa_wcs = wcs_casa2astropy(ia, coords)
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

    cube = SpectralCube.read(path('adv.fits'))

    mask_array = np.array([[True, False], [False, False], [True, True]])
    bool_mask = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                 shape=cube.shape)
    cube = cube.with_mask(bool_mask)

    make_casa_testimage(path('adv.fits'), path('casa.image'))

    if os.path.exists(path('casa.mask')):
        os.system('rm -rf {0}'.format(path('casa.mask')))

    maskpath = os.path.join(path('casa.image'),'casa.mask')
    if os.path.exists(maskpath):
        os.system('rm -rf {0}'.format(maskpath))

    # in this case, casa.mask is the name of the mask, not its path
    make_casa_mask(cube, path('casa.mask'), append_to_image=True,
                   img=path('casa.image'), add_stokes=False,
                   overwrite=True)

    assert os.path.exists(path('casa.image/casa.mask'))

    os.system('rm -rf {0}'.format(path('casa.image')))
    os.system('rm -rf {0}'.format(path('casa.mask')))


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
def test_casa_beams():
    """
    test both ``make_casa_testimage`` and the beam reading tools using casa's
    image reader
    """

    make_casa_testimage(path('adv.fits'), path('casa_adv.image'))
    make_casa_testimage(path('adv_beams.fits'), path('casa_adv_beams.image'))

    cube = SpectralCube.read(path('casa_adv.image'), format='casa_image')

    assert hasattr(cube, 'beam')

    cube_beams = SpectralCube.read(path('casa_adv_beams.image'), format='casa_image')

    assert hasattr(cube_beams, 'beams')
    assert isinstance(cube_beams, VaryingResolutionSpectralCube)

    os.system('rm -rf {0}'.format(path('casa_adv_beams.image')))
