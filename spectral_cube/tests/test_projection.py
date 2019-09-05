from __future__ import print_function, absolute_import, division

import warnings
import pytest
import numpy as np

from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits

from radio_beam import Beam, Beams

from .helpers import assert_allclose
from .test_spectral_cube import cube_and_raw
from ..spectral_cube import SpectralCube
from ..masks import BooleanArrayMask
from ..lower_dimensional_structures import (Projection, Slice, OneDSpectrum,
                                            VaryingResolutionOneDSpectrum)
from ..utils import SliceWarning, WCSCelestialError
from . import path

# set up for parametrization
LDOs = (Projection, Slice, OneDSpectrum)
LDOs_2d = (Projection, Slice,)

two_qty_2d = np.ones((2,2)) * u.Jy
twelve_qty_2d = np.ones((12,12)) * u.Jy
two_qty_1d = np.ones((2,)) * u.Jy
twelve_qty_1d = np.ones((12,)) * u.Jy

data_two = (two_qty_2d, two_qty_2d, two_qty_1d)
data_twelve = (twelve_qty_2d, twelve_qty_2d, twelve_qty_1d)
data_two_2d = (two_qty_2d, two_qty_2d,)
data_twelve_2d = (twelve_qty_2d, twelve_qty_2d,)


def load_projection(filename):

    hdu = fits.open(path(filename))[0]
    proj = Projection.from_hdu(hdu)

    return proj, hdu


@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_two_2d))
def test_slices_of_projections_not_projections(LDO, data):
    # slices of projections that have <2 dimensions should not be projections
    p = LDO(data, copy=False)

    assert not isinstance(p[0,0], LDO)
    assert not isinstance(p[0], LDO)

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_twelve_2d))
def test_copy_false(LDO, data):
    # copy the data so we can manipulate inplace without affecting other tests
    image = data.copy()

    p = LDO(image, copy=False)
    image[3,4] = 2 * u.Jy
    assert_allclose(p[3,4], 2 * u.Jy)

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs, data_twelve))
def test_write(LDO, data, tmpdir):
    p = LDO(data)
    p.write(tmpdir.join('test.fits').strpath)

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_twelve_2d))
def test_preserve_wcs_to(LDO, data):
    # regression for #256
    image = data.copy()
    p = LDO(image, copy=False)
    image[3,4] = 2 * u.Jy

    p2 = p.to(u.mJy)
    assert_allclose(p[3,4], 2 * u.Jy)
    assert_allclose(p[3,4], 2000 * u.mJy)

    assert p2.wcs == p.wcs

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs, data_twelve))
def test_multiplication(LDO, data):
    # regression: 265

    p = LDO(data, copy=False)

    p2 = p * 5

    assert p2.unit == u.Jy
    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value == 5)

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs, data_twelve))
def test_unit_division(LDO, data):
    # regression: 265

    image = data
    p = LDO(image, copy=False)

    p2 = p / u.beam

    assert p2.unit == u.Jy/u.beam
    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_twelve_2d))
def test_isnan(LDO, data):
    # Check that np.isnan strips units

    image = data.copy()
    image[5,6] = np.nan
    p = LDO(image, copy=False)

    mask = np.isnan(p)

    assert mask.sum() == 1
    assert not hasattr(mask, 'unit')

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_twelve_2d))
def test_proj_with_fillvalue(LDO, data):
    # Check that np.isnan strips units

    image = data.copy()
    image[5,6] = np.nan
    p = LDO(image, copy=False)

    p0 = p.with_fill_value(0.)

    assert np.isfinite(p0.filled_data[:]).all()

    mask = np.isnan(p)

    assert mask.sum() == 1
    assert not hasattr(mask, 'unit')

def test_ondespec_with_fillvalue():
    # Check that np.isnan strips units

    spec = twelve_qty_1d.copy()
    spec[5] = np.nan
    p = OneDSpectrum(spec, copy=False)

    p0 = p.with_fill_value(0.)

    assert np.isfinite(p0.filled_data[:]).all()

    mask = np.isnan(p)

    assert mask.sum() == 1
    assert not hasattr(mask, 'unit')

@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs, data_twelve))
def test_self_arith(LDO, data):

    image = data
    p = LDO(image, copy=False)

    p2 = p + p

    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value==2)

    p2 = p - p

    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value==0)


@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs, data_twelve))
def test_self_arith_with_beam(LDO, data):

    exp_beam = Beam(1.0 * u.arcsec)

    image = data
    p = LDO(image, copy=False)
    p = p.with_beam(exp_beam)

    p2 = p + p

    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value==2)
    assert p2.beam == exp_beam

    p2 = p - p

    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value==0)
    assert p2.beam == exp_beam


@pytest.mark.xfail(raises=ValueError, strict=True)
def test_VRODS_wrong_beams_shape():
    '''
    Check that passing Beams with a different shape than the data
    is caught.
    '''
    exp_beams = Beams(np.arange(1, 4) * u.arcsec)

    p = VaryingResolutionOneDSpectrum(twelve_qty_1d, copy=False,
                                      beams=exp_beams)


def test_VRODS_with_beams():

    exp_beams = Beams(np.arange(1, twelve_qty_1d.size + 1) * u.arcsec)

    p = VaryingResolutionOneDSpectrum(twelve_qty_1d, copy=False, beams=exp_beams)
    assert (p.beams == exp_beams).all()

    new_beams = Beams(np.arange(2, twelve_qty_1d.size + 2) * u.arcsec)

    p = p.with_beams(new_beams)
    assert np.all(p.beams == new_beams)


def test_VRODS_slice_with_beams():

    exp_beams = Beams(np.arange(1, twelve_qty_1d.size + 1) * u.arcsec)

    p = VaryingResolutionOneDSpectrum(twelve_qty_1d, copy=False,
                                      wcs=WCS(naxis=1),
                                      beams=exp_beams)

    assert np.all(p[:5].beams == exp_beams[:5])


def test_VRODS_arith_with_beams():

    exp_beams = Beams(np.arange(1, twelve_qty_1d.size + 1) * u.arcsec)

    p = VaryingResolutionOneDSpectrum(twelve_qty_1d, copy=False, beams=exp_beams)

    p2 = p + p

    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value==2)
    assert np.all(p2.beams == exp_beams)

    p2 = p - p

    assert hasattr(p2, '_wcs')
    assert p2.wcs == p.wcs
    assert np.all(p2.value==0)
    assert np.all(p2.beams == exp_beams)


def test_onedspectrum_specaxis_units():

    test_wcs = WCS(naxis=1)
    test_wcs.wcs.cunit = ["m/s"]
    test_wcs.wcs.ctype = ["VELO-LSR"]

    p = OneDSpectrum(twelve_qty_1d, wcs=test_wcs)

    assert p.spectral_axis.unit == u.Unit("m/s")


def test_onedspectrum_with_spectral_unit():

    test_wcs = WCS(naxis=1)
    test_wcs.wcs.cunit = ["m/s"]
    test_wcs.wcs.ctype = ["VELO-LSR"]

    p = OneDSpectrum(twelve_qty_1d, wcs=test_wcs)
    p_new = p.with_spectral_unit(u.km/u.s)

    assert p_new.spectral_axis.unit == u.Unit("km/s")
    np.testing.assert_equal(p_new.spectral_axis.value,
                            1e-3*p.spectral_axis.value)


def test_onedspectrum_input_mask_type():

    test_wcs = WCS(naxis=1)
    test_wcs.wcs.cunit = ["m/s"]
    test_wcs.wcs.ctype = ["VELO-LSR"]

    np_mask = np.ones(twelve_qty_1d.shape, dtype=bool)
    np_mask[1] = False
    bool_mask = BooleanArrayMask(np_mask, wcs=test_wcs,
                                 shape=np_mask.shape)

    # numpy array
    p = OneDSpectrum(twelve_qty_1d, wcs=test_wcs,
                     mask=np_mask)
    assert (p.mask.include() == bool_mask.include()).all()

    # MaskBase
    p = OneDSpectrum(twelve_qty_1d, wcs=test_wcs,
                     mask=bool_mask)
    assert (p.mask.include() == bool_mask.include()).all()

    # No mask
    ones_mask = BooleanArrayMask(np.ones(twelve_qty_1d.shape, dtype=bool),
                                 wcs=test_wcs, shape=np_mask.shape)
    p = OneDSpectrum(twelve_qty_1d, wcs=test_wcs,
                     mask=None)
    assert (p.mask.include() == ones_mask.include()).all()


def test_slice_tricks():
    test_wcs_1 = WCS(naxis=1)
    test_wcs_2 = WCS(naxis=2)

    spec = OneDSpectrum(twelve_qty_1d, wcs=test_wcs_1)
    im = Slice(twelve_qty_2d, wcs=test_wcs_2)

    with warnings.catch_warnings(record=True) as w:
        new = spec[:,None,None] * im[None,:,:]

    assert new.ndim == 3

    # two warnings because we're doing BOTH slices!
    assert len(w) == 2
    assert w[0].category == SliceWarning

    with warnings.catch_warnings(record=True) as w:
        new = spec.array[:,None,None] * im.array[None,:,:]

    assert new.ndim == 3
    assert len(w) == 0

def test_array_property():
    test_wcs_1 = WCS(naxis=1)
    spec = OneDSpectrum(twelve_qty_1d, wcs=test_wcs_1)

    arr = spec.array

    # these are supposed to be the same object, but the 'is' tests fails!
    assert spec.array.data == spec.data

    assert isinstance(arr, np.ndarray)
    assert not isinstance(arr, u.Quantity)

def test_quantity_property():
    test_wcs_1 = WCS(naxis=1)
    spec = OneDSpectrum(twelve_qty_1d, wcs=test_wcs_1)

    arr = spec.quantity

    # these are supposed to be the same object, but the 'is' tests fails!
    assert spec.array.data == spec.data

    assert isinstance(arr, u.Quantity)
    assert not isinstance(arr, OneDSpectrum)


def test_projection_with_beam():

    exp_beam = Beam(1.0 * u.arcsec)

    proj, hdu = load_projection("55.fits")

    # uses from_hdu, which passes beam as kwarg
    assert proj.beam == exp_beam
    assert proj.meta['beam'] == exp_beam

    # load beam from meta
    exp_beam = Beam(1.5 * u.arcsec)

    meta = {"beam": exp_beam}
    new_proj = Projection(hdu.data, wcs=proj.wcs, meta=meta)

    assert new_proj.beam == exp_beam
    assert new_proj.meta['beam'] == exp_beam

    # load beam from given header
    exp_beam = Beam(2.0 * u.arcsec)
    header = hdu.header.copy()
    header = exp_beam.attach_to_header(header)
    new_proj = Projection(hdu.data, wcs=proj.wcs, header=header,
                          read_beam=True)

    assert new_proj.beam == exp_beam
    assert new_proj.meta['beam'] == exp_beam

    # load beam from beam object
    exp_beam = Beam(3.0 * u.arcsec)
    header = hdu.header.copy()
    del header["BMAJ"], header["BMIN"], header["BPA"]
    new_proj = Projection(hdu.data, wcs=proj.wcs, header=header,
                          beam=exp_beam)

    assert new_proj.beam == exp_beam
    assert new_proj.meta['beam'] == exp_beam

    # Slice the projection with a beam and check it's still there
    assert new_proj[:1, :1].beam == exp_beam


def test_ondespectrum_with_beam():

    exp_beam = Beam(1.0 * u.arcsec)

    test_wcs_1 = WCS(naxis=1)
    spec = OneDSpectrum(twelve_qty_1d, wcs=test_wcs_1)

    # load beam from meta
    meta = {"beam": exp_beam}
    new_spec = OneDSpectrum(spec.data, wcs=spec.wcs, meta=meta)

    assert new_spec.beam == exp_beam
    assert new_spec.meta['beam'] == exp_beam

    # load beam from given header
    hdu = spec.hdu
    exp_beam = Beam(2.0 * u.arcsec)
    header = hdu.header.copy()
    header = exp_beam.attach_to_header(header)
    new_spec = OneDSpectrum(hdu.data, wcs=spec.wcs, header=header,
                            read_beam=True)

    assert new_spec.beam == exp_beam
    assert new_spec.meta['beam'] == exp_beam

    # load beam from beam object
    exp_beam = Beam(3.0 * u.arcsec)
    header = hdu.header.copy()
    new_spec = OneDSpectrum(hdu.data, wcs=spec.wcs, header=header,
                            beam=exp_beam)

    assert new_spec.beam == exp_beam
    assert new_spec.meta['beam'] == exp_beam

    # Slice the spectrum with a beam and check it's still there
    assert new_spec[:1].beam == exp_beam


@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs, data_twelve))
def test_ldo_attach_beam(LDO, data):

    exp_beam = Beam(1.0 * u.arcsec)
    newbeam = Beam(2.0 * u.arcsec)

    p = LDO(data, copy=False, beam=exp_beam)

    new_p = p.with_beam(newbeam)

    assert p.beam == exp_beam
    assert p.meta['beam'] == exp_beam

    assert new_p.beam == newbeam
    assert new_p.meta['beam'] == newbeam


@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_two_2d))
def test_projection_from_hdu(LDO, data):

    p = LDO(data, copy=False)

    hdu = p.hdu

    p_new = LDO.from_hdu(hdu)

    assert (p == p_new).all()


def test_projection_subimage():

    proj, hdu = load_projection("55.fits")

    proj1 = proj.subimage(xlo=1, xhi=3)
    proj2 = proj.subimage(xlo=24.06269 * u.deg,
                          xhi=24.06206 * u.deg)

    assert proj1.shape == (5, 2)
    assert proj2.shape == (5, 2)
    assert proj1.wcs.wcs.compare(proj2.wcs.wcs)
    assert proj.beam == proj1.beam
    assert proj.beam == proj2.beam

    proj3 = proj.subimage(ylo=1, yhi=3)
    proj4 = proj.subimage(ylo=29.93464 * u.deg,
                          yhi=29.93522 * u.deg)

    assert proj3.shape == (2, 5)
    assert proj4.shape == (2, 5)
    assert proj3.wcs.wcs.compare(proj4.wcs.wcs)

    proj5 = proj.subimage()

    assert proj5.shape == proj.shape
    assert proj5.wcs.wcs.compare(proj.wcs.wcs)
    assert np.all(proj5.value == proj.value)


def test_projection_subimage_nocelestial_fail():

    cube, data = cube_and_raw('255_delta.fits')

    proj = cube.moment0(axis=1)

    with pytest.raises(WCSCelestialError) as exc:
        proj.subimage(xlo=1, xhi=3)

    assert exc.value.args[0] == ("WCS does not contain two spatial axes.")


@pytest.mark.parametrize('LDO', LDOs_2d)
def test_twod_input_mask_type(LDO):

    test_wcs = WCS(naxis=2)
    test_wcs.wcs.cunit = ["deg", "deg"]
    test_wcs.wcs.ctype = ["RA---SIN", 'DEC--SIN']

    np_mask = np.ones(twelve_qty_2d.shape, dtype=bool)
    np_mask[1] = False
    bool_mask = BooleanArrayMask(np_mask, wcs=test_wcs,
                                 shape=np_mask.shape)

    # numpy array
    p = LDO(twelve_qty_2d, wcs=test_wcs,
            mask=np_mask)
    assert (p.mask.include() == bool_mask.include()).all()

    # MaskBase
    p = LDO(twelve_qty_2d, wcs=test_wcs,
            mask=bool_mask)
    assert (p.mask.include() == bool_mask.include()).all()

    # No mask
    ones_mask = BooleanArrayMask(np.ones(twelve_qty_2d.shape, dtype=bool),
                                 wcs=test_wcs, shape=np_mask.shape)
    p = LDO(twelve_qty_2d, wcs=test_wcs,
            mask=None)
    assert (p.mask.include() == ones_mask.include()).all()


@pytest.mark.xfail
def test_mask_convolve():
    # Numpy is fundamentally incompatible with the objects we have created.
    # np.ma.is_masked(array) checks specifically for the array's _mask
    # attribute.  We would have to refactor deeply to correct this, and I
    # really don't want to do that because 'None' is a much more reasonable
    # and less dangerous default for a mask.
    test_wcs_1 = WCS(naxis=1)
    spec = OneDSpectrum(twelve_qty_1d, wcs=test_wcs_1)

    assert spec.mask is False

    from astropy.convolution import convolve,Box1DKernel
    convolve(spec, Box1DKernel(3))

def test_convolve():
    test_wcs_1 = WCS(naxis=1)
    spec = OneDSpectrum(twelve_qty_1d, wcs=test_wcs_1)

    from astropy.convolution import Box1DKernel
    specsmooth = spec.spectral_smooth(Box1DKernel(1))

    np.testing.assert_allclose(spec, specsmooth)

def test_spectral_interpolate():
    test_wcs_1 = WCS(naxis=1)
    test_wcs_1.wcs.cunit[0] = 'GHz'
    spec = OneDSpectrum(np.arange(12)*u.Jy, wcs=test_wcs_1)

    new_xaxis = test_wcs_1.wcs_pix2world(np.linspace(0,11,23), 0)[0] * u.Unit(test_wcs_1.wcs.cunit[0])
    new_spec = spec.spectral_interpolate(new_xaxis)

    np.testing.assert_allclose(new_spec, np.linspace(0,11,23)*u.Jy)


def test_spectral_interpolate_with_mask():

    hdu = fits.open(path("522_delta.fits"))[0]

    # Swap the velocity axis so indiff < 0 in spectral_interpolate
    hdu.header["CDELT3"] = - hdu.header["CDELT3"]

    cube = SpectralCube.read(hdu)

    mask = np.ones(cube.shape, dtype=bool)
    mask[:2] = False

    masked_cube = cube.with_mask(mask)

    spec = masked_cube[:, 0, 0]

    # midpoint between each position
    sg = (spec.spectral_axis[1:] + spec.spectral_axis[:-1])/2.

    result = spec.spectral_interpolate(spectral_grid=sg[::-1])

    # The output makes CDELT3 > 0 (reversed spectral axis) so the masked
    # portion are the final 2 channels.
    np.testing.assert_almost_equal(result.filled_data[:].value,
                                   [0.0, 0.5, np.NaN, np.NaN])

def test_spectral_interpolate_reversed():

    cube, data = cube_and_raw('522_delta.fits')

    # Reverse spectral axis
    sg = cube.spectral_axis[::-1]

    spec = cube[:, 0, 0]

    result = spec.spectral_interpolate(spectral_grid=sg)

    np.testing.assert_almost_equal(sg.value, result.spectral_axis.value)

def test_spectral_interpolate_with_fillvalue():

    cube, data = cube_and_raw('522_delta.fits')

    # Step one channel out of bounds.
    sg = ((cube.spectral_axis[0]) -
          (cube.spectral_axis[1] - cube.spectral_axis[0]) *
          np.linspace(1,4,4))

    spec = cube[:, 0, 0]

    result = spec.spectral_interpolate(spectral_grid=sg,
                                       fill_value=42)
    np.testing.assert_almost_equal(result.value,
                                   np.ones(4)*42)


def test_spectral_units():
    # regression test for issue 391

    cube, data = cube_and_raw('255_delta.fits')

    sp = cube[:,0,0]

    assert sp.spectral_axis.unit == u.km/u.s
    assert sp.header['CUNIT1'] == 'km s-1'

    sp = cube.with_spectral_unit(u.m/u.s)[:,0,0]

    assert sp.spectral_axis.unit == u.m/u.s
    assert sp.header['CUNIT1'] in ('m s-1', 'm/s')

def test_repr_1d():

    cube, data = cube_and_raw('255_delta.fits')

    sp = cube[:,0,0]

    print(sp)
    print(sp[1:-1])

    assert 'OneDSpectrum' in sp.__repr__()
    assert 'OneDSpectrum' in sp[1:-1].__repr__()

def test_1d_slices():

    cube, data = cube_and_raw('255_delta.fits')

    sp = cube[:,0,0]

    assert sp.max() == cube.max(axis=0)[0,0]
    assert not isinstance(sp.max(), OneDSpectrum)

    sp = cube[:-1,0,0]

    assert sp.max() == cube[:-1,:,:].max(axis=0)[0,0]
    assert not isinstance(sp.max(), OneDSpectrum)

@pytest.mark.parametrize('method',
                         ('min', 'max', 'std', 'mean', 'sum', 'cumsum',
                          'nansum', 'ptp', 'var'),
                        )
def test_1d_slice_reductions(method):

    cube, data = cube_and_raw('255_delta.fits')

    sp = cube[:,0,0]

    if hasattr(cube, method):
        assert getattr(sp, method)() == getattr(cube, method)(axis=0)[0,0]
    else:
        getattr(sp, method)()

    assert hasattr(sp, '_fill_value')

    assert 'OneDSpectrum' in sp.__repr__()
    assert 'OneDSpectrum' in sp[1:-1].__repr__()


def test_1d_slice_round():
    cube, data = cube_and_raw('255_delta.fits')

    sp = cube[:,0,0]

    assert all(sp.value.round() == sp.round().value)

    assert hasattr(sp, '_fill_value')
    assert hasattr(sp.round(), '_fill_value')

    assert 'OneDSpectrum' in sp.round().__repr__()
    assert 'OneDSpectrum' in sp[1:-1].round().__repr__()

def test_LDO_arithmetic():
    cube, data = cube_and_raw('vda.fits')

    sp = cube[:,0,0]

    spx2 = sp * 2
    assert np.all(spx2.value == sp.value*2)
    assert np.all(spx2.filled_data[:].value == sp.value*2)


def test_beam_jtok_2D():

    cube, data = cube_and_raw('advs.fits')
    cube._meta['BUNIT'] = 'Jy / beam'
    cube._unit = u.Jy / u.beam

    plane = cube[0]

    freq = cube.with_spectral_unit(u.GHz).spectral_axis[0]

    equiv = plane.beam.jtok_equiv(freq)
    jtok = plane.beam.jtok(freq)

    Kplane = plane.to(u.K, equivalencies=equiv, freq=freq)
    np.testing.assert_almost_equal(Kplane.value,
                                   (plane.value * jtok).value)

    # test that the beam equivalencies are correctly automatically defined
    Kplane = plane.to(u.K, freq=freq)
    np.testing.assert_almost_equal(Kplane.value,
                                   (plane.value * jtok).value)

def test_basic_arrayness():
    cube, data = cube_and_raw('adv.fits')

    assert cube.shape == data.shape

    spec = cube[:,0,0]

    assert np.all(np.asanyarray(spec).value == data[:,0,0])
    assert np.all(np.array(spec) == data[:,0,0])
    assert np.all(np.asarray(spec) == data[:,0,0])
    # These are commented out because it is presently not possible to convert
    # projections to masked arrays
    # assert np.all(np.ma.asanyarray(spec).value == data[:,0,0])
    # assert np.all(np.ma.asarray(spec) == data[:,0,0])
    # assert np.all(np.ma.array(spec) == data[:,0,0])

    slc = cube[0,:,:]

    assert np.all(np.asanyarray(slc).value == data[0,:,:])
    assert np.all(np.array(slc) == data[0,:,:])
    assert np.all(np.asarray(slc) == data[0,:,:])
    # assert np.all(np.ma.asanyarray(slc).value == data[0,:,:])
    # assert np.all(np.ma.asarray(slc) == data[0,:,:])
    # assert np.all(np.ma.array(slc) == data[0,:,:])


def test_spatial_world_extrema_2D():

    hdu = fits.open(path("522_delta.fits"))[0]

    cube = SpectralCube.read(hdu)

    plane = cube[0]

    assert (cube.world_extrema == plane.world_extrema).all()
    assert (cube.longitude_extrema == plane.longitude_extrema).all()
    assert (cube.latitude_extrema == plane.latitude_extrema).all()
