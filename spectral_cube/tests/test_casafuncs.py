from __future__ import print_function, absolute_import, division

import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

from astropy import units as u

from ..io.casa_masks import make_casa_mask
from ..io.casa_wcs import wcs_casa2astropy
from .. import SpectralCube, StokesSpectralCube, BooleanArrayMask, VaryingResolutionSpectralCube

try:
    import casatools
    from casatools import image
    CASA_INSTALLED = True
except ImportError:
    try:
        from taskinit import ia as image
        CASA_INSTALLED = True
    except ImportError:
        CASA_INSTALLED = False


def make_casa_testimage(infile, outname):

    infile = str(infile)
    outname = str(outname)

    if not CASA_INSTALLED:
        raise Exception("Attempted to make a CASA test image in a non-CASA "
                        "environment")

    ia = image()

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


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
@pytest.mark.parametrize('filename', ('data_adv', 'data_advs', 'data_sdav',
                                      'data_vad', 'data_vsad'),
                         indirect=['filename'])
def test_casa_read(filename, tmp_path):

    # Check that SpectralCube.read returns data with the same shape and values
    # if read from CASA as if read from FITS.

    cube = SpectralCube.read(filename)

    make_casa_testimage(filename, tmp_path / 'casa.image')

    casacube = SpectralCube.read(tmp_path / 'casa.image')

    assert casacube.shape == cube.shape
    assert_allclose(casacube.unmasked_data[:].value,
                    cube.unmasked_data[:].value)


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
def test_casa_read_stokes(data_advs, tmp_path):

    # Check that StokesSpectralCube.read returns data with the same shape and values
    # if read from CASA as if read from FITS.

    cube = StokesSpectralCube.read(data_advs)

    make_casa_testimage(data_advs, tmp_path / 'casa.image')

    casacube = StokesSpectralCube.read(tmp_path / 'casa.image')

    assert casacube.I.shape == cube.I.shape
    assert_allclose(casacube.I.unmasked_data[:].value,
                    cube.I.unmasked_data[:].value)


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
def test_casa_mask(data_adv, tmp_path):

    # This tests the make_casa_mask function which can be used to create a mask
    # file in an existing image.

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
    casa_wcs = wcs_casa2astropy(coords.naxes(), coords.torecord())
    header = casa_wcs.to_header()  # Invokes transform

    # Compare some basic properties EXCLUDING the spectral axis
    assert np.allclose(cube.wcs.wcs.crval[:2], casa_wcs.wcs.crval[:2])
    assert np.all(cube.wcs.wcs.cdelt[:2] == casa_wcs.wcs.cdelt[:2])
    assert np.all(list(cube.wcs.wcs.cunit)[:2] == list(casa_wcs.wcs.cunit)[:2])
    assert np.all(list(cube.wcs.wcs.ctype)[:2] == list(casa_wcs.wcs.ctype)[:2])

    # Reference pixels in CASA are 1 pixel off.
    assert_allclose(cube.wcs.wcs.crpix, casa_wcs.wcs.crpix,
                    atol=1.0)


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
def test_casa_mask_append(data_adv, tmp_path):

    # This tests the append option for the make_casa_mask function

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


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
def test_casa_beams(data_adv, data_adv_beams, tmp_path):

    # Test both make_casa_testimage and the beam reading tools using casa's
    # image reader

    make_casa_testimage(data_adv, tmp_path / 'casa_adv.image')
    make_casa_testimage(data_adv_beams, tmp_path / 'casa_adv_beams.image')

    cube = SpectralCube.read(tmp_path / 'casa_adv.image', format='casa_image')

    assert hasattr(cube, 'beam')

    cube_beams = SpectralCube.read(tmp_path / 'casa_adv_beams.image', format='casa_image')

    assert hasattr(cube_beams, 'beams')
    assert isinstance(cube_beams, VaryingResolutionSpectralCube)
