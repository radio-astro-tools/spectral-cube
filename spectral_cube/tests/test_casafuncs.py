from __future__ import print_function, absolute_import, division

from itertools import product
import pytest
import numpy as np
from numpy.testing import assert_allclose
import os

from astropy import units as u

from ..io.casa_masks import make_casa_mask
from ..io.casa_image import wcs_casa2astropy, casa_image_dask_reader
from .. import SpectralCube, StokesSpectralCube, BooleanArrayMask, VaryingResolutionSpectralCube
from . import path

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


# NOTE: the (127, 337, 109) example is to make sure that things work correctly
# when the shape isn't a mulitple of the chunk size along any
# dimension.
SHAPES = [(3, 4, 5), (129, 128, 130), (513, 128, 128), (128, 513, 128),
          (128, 128, 513), (512, 64, 64), (127, 337, 109)]


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
@pytest.mark.parametrize(('memmap', 'shape'), product([False, True], SHAPES))
def test_casa_image_dask_reader(tmpdir, memmap, shape):

    # Unit tests for the low-level casa_image_dask_reader function which can
    # read a CASA image or mask to a Dask array.

    reference = np.random.random(shape).astype(np.float32)

    # CASA seems to have precision issues when computing masks with values
    # very close to e.g. 0.5 in >0.5. To avoid this, we filter out random
    # values close to the boundaries that we use below.
    reference[np.isclose(reference, 0.2)] += 0.05
    reference[np.isclose(reference, 0.5)] += 0.05
    reference[np.isclose(reference, 0.8)] += 0.05

    os.chdir(tmpdir.strpath)

    # Start off with a simple example with no mask. Note that CASA requires
    # the array to be transposed in order to match what we would expect.

    ia = image()
    ia.fromarray('basic.image', pixels=reference.T, log=False)
    ia.close()

    array1 = casa_image_dask_reader('basic.image', memmap=memmap)
    assert array1.dtype == np.float32
    assert_allclose(array1, reference)

    # Check slicing
    assert_allclose(array1[:2, :1, :3], reference[:2, :1, :3])

    # Try and get a mask - this should fail since there isn't one.

    with pytest.raises(FileNotFoundError):
        casa_image_dask_reader('basic.image', mask=True, memmap=memmap)

    # Now create an array with a simple uniform mask.

    ia = image()
    ia.fromarray('scalar_mask.image', pixels=reference.T, log=False)
    ia.calcmask(mask='T')
    ia.close()

    array2 = casa_image_dask_reader('scalar_mask.image', memmap=memmap)
    assert_allclose(array2, reference)

    mask2 = casa_image_dask_reader('scalar_mask.image', mask=True, memmap=memmap)
    assert mask2.dtype is np.dtype('bool')
    assert mask2.shape == array2.shape
    assert np.all(mask2)

    # Check with a full 3-d mask

    ia = image()
    ia.fromarray('array_mask.image', pixels=reference.T, log=False)
    ia.calcmask(mask='array_mask.image>0.5')
    ia.close()

    array3 = casa_image_dask_reader('array_mask.image', memmap=memmap)
    assert_allclose(array3, reference)

    mask3 = casa_image_dask_reader('array_mask.image', mask=True, memmap=memmap)
    keep = mask3 != (reference > 0.5)
    assert_allclose(mask3, reference > 0.5)

    # Check slicing
    assert_allclose(mask3[:2, :1, :3], (reference > 0.5)[:2, :1, :3])

    # Test specifying the mask name

    ia = image()
    ia.fromarray('array_masks.image', pixels=reference.T, log=False)
    ia.calcmask(mask='array_masks.image>0.5')
    ia.calcmask(mask='array_masks.image>0.2')
    ia.calcmask(mask='array_masks.image>0.8', name='gt08')
    ia.close()

    array4 = casa_image_dask_reader('array_masks.image', memmap=memmap)
    assert_allclose(array4, reference)

    mask4 = casa_image_dask_reader('array_masks.image', mask=True, memmap=memmap)
    assert_allclose(mask4, reference > 0.5)

    mask5 = casa_image_dask_reader('array_masks.image', mask='mask0', memmap=memmap)
    assert_allclose(mask5, reference > 0.5)

    mask6 = casa_image_dask_reader('array_masks.image', mask='mask1', memmap=memmap)
    assert_allclose(mask6, reference > 0.2)

    mask7 = casa_image_dask_reader('array_masks.image', mask='gt08', memmap=memmap)
    assert_allclose(mask7, reference > 0.8)

    # Check that things still work if we write the array out with doubles

    reference = np.random.random(shape).astype(np.float64)

    ia = image()
    ia.fromarray('double.image', pixels=reference.T, type='d', log=False)
    ia.close()

    array8 = casa_image_dask_reader('double.image', memmap=memmap)
    assert array8.dtype == np.float64
    assert_allclose(array8, reference)
