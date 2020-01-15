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
    casaOK = True
except ImportError:
    try:
        from taskinit import ia
        casaOK = True
    except ImportError:
        print("Run in CASA environment.")
        casaOK = False


def make_casa_testimage(infile, outname):

    infile = str(infile)
    outname = str(outname)

    if not casaOK:
        raise Exception("Attempted to make a CASA test image in a non-CASA "
                        "environment")

    ia = casatools.image()

    ia.fromfits(infile=infile, outfile=outname, overwrite=True)
    ia.unlock()
    ia.close()
    ia.done()

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

        ia.unlock()
        ia.close()
        ia.done()


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
@pytest.mark.parametrize('filename', ('data_adv', 'data_advs', 'data_sdav',
                                      'data_vad', 'data_vsad'),
                         indirect=['filename'])
def test_casa_read(filename, tmp_path):

    cube = SpectralCube.read(filename)

    make_casa_testimage(filename, tmp_path / 'casa.image')

    casacube = SpectralCube.read(tmp_path / 'casa.image', format='casa_image')

    assert casacube.shape == cube.shape
    # what other equalities should we check?


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
def test_casa_mask(data_adv, tmp_path):

    cube = SpectralCube.read(data_adv)

    mask_array = np.array([[True, False], [False, False], [True, True]])
    bool_mask = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                 shape=cube.shape)
    cube = cube.with_mask(bool_mask)

    make_casa_mask(cube, str(tmp_path / 'casa.mask'), add_stokes=False,
                   append_to_image=False, overwrite=True)

    ia = casatools.image()

    ia.open(str(tmp_path / 'casa.mask'))

    casa_mask = ia.getchunk()

    coords = ia.coordsys()

    ia.unlock()
    ia.close()
    ia.done()

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
def test_casa_mask_append(data_adv, tmp_path):

    cube = SpectralCube.read(data_adv)

    mask_array = np.array([[True, False], [False, False], [True, True]])
    bool_mask = BooleanArrayMask(mask=mask_array, wcs=cube._wcs,
                                 shape=cube.shape)
    cube = cube.with_mask(bool_mask)

    make_casa_testimage(data_adv, tmp_path / 'casa.image')

    # in this case, casa.mask is the name of the mask, not its path
    make_casa_mask(cube, 'casa.mask', append_to_image=True,
                   img=str(tmp_path / 'casa.image'), add_stokes=False, overwrite=True)

    assert os.path.exists(tmp_path / 'casa.image/casa.mask')


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
def test_casa_beams(data_adv, data_adv_beams, tmp_path):
    """
    test both ``make_casa_testimage`` and the beam reading tools using casa's
    image reader
    """

    make_casa_testimage(data_adv, tmp_path / 'casa_adv.image')
    make_casa_testimage(data_adv_beams, tmp_path / 'casa_adv_beams.image')

    cube = SpectralCube.read(tmp_path / 'casa_adv.image', format='casa_image')

    assert hasattr(cube, 'beam')

    cube_beams = SpectralCube.read(tmp_path / 'casa_adv_beams.image', format='casa_image')

    assert hasattr(cube_beams, 'beams')
    assert isinstance(cube_beams, VaryingResolutionSpectralCube)


@pytest.mark.skipif(not casaOK, reason='CASA tests must be run in a CASA environment.')
def test_casa_arrayslicer(data_adv, tmp_path):
    """
    """

    make_casa_testimage(data_adv, tmp_path / 'casa_adv.image')

    from spectral_cube.io.casa_image import ArraylikeCasaData

    arr = ArraylikeCasaData(str(tmp_path / 'casa_adv.image'))

    assert arr.shape == (4,3,2)
    assert arr.ndim == 3
    assert arr.dtype == np.float64

    assert arr[:,0,0].size == 4
    assert arr[:,:,0].shape == (4,3)

    assert arr[:3,:2,:1].shape == (3,2,1)
    assert arr[:3,:2,0].shape == (3,2,)
    assert arr[:3,:2,1].shape == (3,2,)

    cube = SpectralCube.read(tmp_path / 'casa_adv.image', format='casa_image')

    assert cube[:3,:2,:1].shape == (3,2,1)
    assert np.array(cube.filled_data[:3,:2,:1]).shape == (3,2,1)
    assert np.array(cube[:3,:2,:1].mask.include()).shape == (3,2,1)
