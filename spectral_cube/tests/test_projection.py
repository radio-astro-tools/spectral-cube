from __future__ import print_function, absolute_import, division

import warnings
import pytest
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits

from .helpers import assert_allclose
from .test_spectral_cube import cube_and_raw
from ..lower_dimensional_structures import Projection, Slice, OneDSpectrum
from ..utils import SliceWarning, WCSCelestialError
from . import path

try:
    from radio_beam import Beam
    RADIO_BEAM_INSTALLED = True
except ImportError:
    RADIO_BEAM_INSTALLED = False

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


@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_two_2d))
def test_projection_from_hdu(LDO, data):

    p = LDO(data, copy=False)

    hdu = p.hdu

    p_new = LDO.from_hdu(hdu)

    assert (p == p_new).all()


@pytest.mark.skipif('not RADIO_BEAM_INSTALLED')
@pytest.mark.parametrize(('LDO', 'data'),
                         zip(LDOs_2d, data_two_2d))
def test_projection_from_hdu_with_beam(LDO, data):

    p = LDO(data, copy=False)

    hdu = p.hdu

    beam = Beam(1 * u.arcsec)
    hdu.header = beam.attach_to_header(hdu.header)

    p_new = LDO.from_hdu(hdu)

    assert (p == p_new).all()
    assert beam == p_new.meta['beam']


def test_projection_subimage():

    proj, hdu = load_projection("55.fits")

    proj1 = proj.subimage(xlo=1, xhi=3)
    proj2 = proj.subimage(xlo=24.06269 * u.deg,
                          xhi=24.06206 * u.deg)

    assert proj1.shape == (5, 2)
    assert proj2.shape == (5, 2)
    assert proj1.wcs.wcs.compare(proj2.wcs.wcs)

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

