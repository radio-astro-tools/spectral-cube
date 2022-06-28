from __future__ import print_function, absolute_import, division

import os
import shutil
from itertools import product

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as u
from casa_formats_io import coordsys_to_astropy_wcs

from ..io.casa_masks import make_casa_mask
from .. import StokesSpectralCube, BooleanArrayMask

from .. import SpectralCube, VaryingResolutionSpectralCube

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

DATA = os.path.join(os.path.dirname(__file__), 'data')


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

def make_casa_stokes_testimage(infile, outname):

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

    cube = StokesSpectralCube.read(infile)
    if isinstance(cube.I, VaryingResolutionSpectralCube):
        ia.open(outname)

        # populate restoring beam emptily
        ia.setrestoringbeam(remove=True)

        for polnum, comp in enumerate(cube.components):

            for channum, beam in enumerate(cube[comp].beams):
                casabdict = {'major': {'value':beam.major.to(u.deg).value, 'unit':'deg'},
                            'minor': {'value':beam.minor.to(u.deg).value, 'unit':'deg'},
                            'positionangle': {'value':beam.pa.to(u.deg).value, 'unit':'deg'}
                            }

                ia.setrestoringbeam(beam=casabdict, channel=channum, polarization=polnum)

        ia.unlock()
        ia.close()
        ia.done()


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(('memmap', 'bigendian'), product((False, True), (False, True)))
def test_casa_read_basic(memmap, bigendian):

    # Check that SpectralCube.read works for an example CASA dataset stored
    # in the tests directory. This test should NOT require CASA, whereas a
    # number of tests below require CASA to generate test datasets. The present
    # test is to ensure CASA is not required for reading.

    if bigendian:
        cube = SpectralCube.read(os.path.join(DATA, 'basic_bigendian.image'), memmap=memmap)
    else:
        cube = SpectralCube.read(os.path.join(DATA, 'basic.image'), memmap=memmap)

    assert cube.shape == (3, 4, 5)
    assert_allclose(cube.wcs.pixel_to_world_values(1, 2, 3),
                    [2.406271e+01, 2.993521e+01, 1.421911e+09])

    # Carry out an operation to make sure the underlying data array works

    cube.moment0()

    # Slice the dataset

    assert_quantity_allclose(cube.unmasked_data[0, 0, :],
                             [1, 1, 1, 1, 1] * u.Jy / u.beam)
    assert_quantity_allclose(cube.unmasked_data[0, 1, 2], 1 * u.Jy / u.beam)


def test_casa_read_basic_nodask():

    # For CASA datasets, the default when reading cubes is use_dask=True.
    # Here we check that setting use_dask=False explicitly raises an error.

    with pytest.raises(ValueError, match='Loading CASA datasets is not possible with use_dask=False'):
        SpectralCube.read(os.path.join(DATA, 'basic.image'), use_dask=False)


def test_casa_read_basic_nomask():

    # Make sure things work well if there is no mask in the data

    cube = SpectralCube.read(os.path.join(DATA, 'nomask.image'))
    assert cube.shape == (3, 4, 5)
    assert_allclose(cube.wcs.pixel_to_world_values(1, 2, 3),
                    [2.406271e+01, 2.993521e+01, 1.421911e+09])

    # Carry out an operation to make sure the underlying data array works

    cube.moment0()

    # Slice the dataset

    assert_quantity_allclose(cube.unmasked_data[0, 0, :],
                             [1, 1, 1, 1, 1] * u.Jy / u.beam)
    assert_quantity_allclose(cube.unmasked_data[0, 1, 2], 1 * u.Jy / u.beam)

    # Slice the cube

    assert_quantity_allclose(cube[:, 0, 0],
                             [1, 1, 1] * u.Jy / u.beam)


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
@pytest.mark.parametrize('filename', ('data_adv', 'data_advs', 'data_sdav',
                                      'data_vad', 'data_vsad'),
                         indirect=['filename'])
def test_casa_read_nomask(filename, tmp_path):

    # As for test_casa_read, but we remove the mask to make sure
    # that we can still read in the cubes

    cube = SpectralCube.read(filename)

    make_casa_testimage(filename, tmp_path / 'casa.image')
    shutil.rmtree(tmp_path / 'casa.image' / 'mask0')

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
    casa_wcs = coordsys_to_astropy_wcs(coords.torecord())
    header = casa_wcs.to_header()  # Invokes transform

    # Compare some basic properties EXCLUDING the spectral axis
    assert_allclose(cube.wcs.wcs.crval[:2], casa_wcs.wcs.crval[:2])
    assert_allclose(cube.wcs.wcs.cdelt[:2], casa_wcs.wcs.cdelt[:2])
    assert np.all(list(cube.wcs.wcs.cunit)[:2] == list(casa_wcs.wcs.cunit)[:2])
    assert np.all(list(cube.wcs.wcs.ctype)[:2] == list(casa_wcs.wcs.ctype)[:2])

    assert_allclose(cube.wcs.wcs.crpix, casa_wcs.wcs.crpix)


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


@pytest.mark.skipif(not CASA_INSTALLED, reason='CASA tests must be run in a CASA environment.')
def test_casa_beams_stokes(data_advs_beams_fullstokes, tmp_path):
    '''
    Varying resolution spectral-cube with full Stokes.
    '''

    # Test both make_casa_testimage and the beam reading tools using casa's
    # image reader

    cube = StokesSpectralCube.read(data_advs_beams_fullstokes)

    make_casa_stokes_testimage(data_advs_beams_fullstokes,
                               tmp_path / 'casa_adv_beams_stokes.image')

    casacube = StokesSpectralCube.read(tmp_path / 'casa_adv_beams_stokes.image', format='casa_image')

    for component in 'IQUV':

        cube_component = getattr(cube, component)
        casacube_component = getattr(casacube, component)

        assert casacube_component.shape == cube_component.shape
        assert_allclose(casacube_component.unmasked_data[:].value,
                        cube_component.unmasked_data[:].value)

        assert casacube_component.shape == cube_component.shape
        assert_allclose(casacube_component.unmasked_data[:].value,
                        cube_component.unmasked_data[:].value)

        assert casacube_component.beams == cube_component.beams

        assert isinstance(casacube_component, VaryingResolutionSpectralCube)
