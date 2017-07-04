from __future__ import print_function, absolute_import, division

from astropy.io import fits

import pytest

from ..wcs_utils import *
from . import path


def test_wcs_dropping():
    wcs = WCS(naxis=4)
    wcs.wcs.pc = np.zeros([4, 4])
    np.fill_diagonal(wcs.wcs.pc, np.arange(1, 5))
    pc = wcs.wcs.pc  # for later use below

    dropped = drop_axis(wcs, 0)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([2, 3, 4]))
    dropped = drop_axis(wcs, 1)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1, 3, 4]))
    dropped = drop_axis(wcs, 2)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1, 2, 4]))
    dropped = drop_axis(wcs, 3)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1, 2, 3]))

    wcs = WCS(naxis=4)
    wcs.wcs.cd = pc

    dropped = drop_axis(wcs, 0)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([2, 3, 4]))
    dropped = drop_axis(wcs, 1)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1, 3, 4]))
    dropped = drop_axis(wcs, 2)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1, 2, 4]))
    dropped = drop_axis(wcs, 3)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1, 2, 3]))


def test_wcs_swapping():
    wcs = WCS(naxis=4)
    wcs.wcs.pc = np.zeros([4, 4])
    np.fill_diagonal(wcs.wcs.pc, np.arange(1, 5))
    pc = wcs.wcs.pc  # for later use below

    swapped = wcs_swapaxes(wcs, 0, 1)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([2, 1, 3, 4]))
    swapped = wcs_swapaxes(wcs, 0, 3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([4, 2, 3, 1]))
    swapped = wcs_swapaxes(wcs, 2, 3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([1, 2, 4, 3]))

    wcs = WCS(naxis=4)
    wcs.wcs.cd = pc

    swapped = wcs_swapaxes(wcs, 0, 1)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([2, 1, 3, 4]))
    swapped = wcs_swapaxes(wcs, 0, 3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([4, 2, 3, 1]))
    swapped = wcs_swapaxes(wcs, 2, 3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([1, 2, 4, 3]))


def test_add_stokes():
    wcs = WCS(naxis=3)

    for ii in range(4):
        outwcs = add_stokes_axis_to_wcs(wcs, ii)
        assert outwcs.wcs.naxis == 4


def test_axis_names():
    wcs = WCS(path('adv.fits'))
    assert axis_names(wcs) == ['RA', 'DEC', 'VOPT']
    wcs = WCS(path('vad.fits'))
    assert axis_names(wcs) == ['VOPT', 'RA', 'DEC']


def test_wcs_slice():
    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [50., 45., 30.]
    wcs_new = slice_wcs(wcs, (slice(10,20), slice(None), slice(20,30)))
    np.testing.assert_allclose(wcs_new.wcs.crpix, [30., 45., 20.])

def test_wcs_slice_reversal():
    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [50., 45., 30.]
    wcs.wcs.crval = [0., 0., 0.]
    wcs.wcs.cdelt = [1., 1., 1.]
    wcs_new = slice_wcs(wcs, (slice(None, None, -1), slice(None), slice(None)),
                        shape=[100., 150., 200.])
    spaxis = wcs.sub([0]).wcs_pix2world(np.arange(100), 0)
    new_spaxis = wcs_new.sub([0]).wcs_pix2world(np.arange(100), 0)

    np.testing.assert_allclose(spaxis, new_spaxis[::-1])


def test_wcs_comparison():
    wcs1 = WCS(naxis=3)
    wcs1.wcs.crpix = np.array([50., 45., 30.], dtype='float32')
    
    wcs2 = WCS(naxis=3)
    wcs2.wcs.crpix = np.array([50., 45., 30.], dtype='float64')

    wcs3 = WCS(naxis=3)
    wcs3.wcs.crpix = np.array([50., 45., 31.], dtype='float64')

    wcs4 = WCS(naxis=3)
    wcs4.wcs.crpix = np.array([50., 45., 30.0001], dtype='float64')

    assert check_equality(wcs1,wcs2)
    assert not check_equality(wcs1,wcs3)
    assert check_equality(wcs1, wcs3, wcs_tolerance=1.0e1)
    assert not check_equality(wcs1,wcs4)
    assert check_equality(wcs1, wcs4, wcs_tolerance=1e-3)

def test_strip_wcs():

    header1 = fits.Header.fromtextfile(path('cubewcs1.hdr'))
    header1_stripped = strip_wcs_from_header(header1)

    with open(path('cubewcs1.hdr'),'r') as fh:
        hdrlines = fh.readlines()

    hdrlines.insert(-20,"\n")
    hdrlines.insert(-1,"\n")
    with open(path('cubewcs1_blanks.hdr'),'w') as fh:
        fh.writelines(hdrlines)

    header2 = fits.Header.fromtextfile(path('cubewcs1_blanks.hdr'))
    header2_stripped = strip_wcs_from_header(header2)

    assert header1_stripped == header2_stripped

@pytest.mark.parametrize(('position', 'result'),
                         (('start', 0.),
                          ('middle', 5e-5),
                          ('end', 10e-5)))
def test_drop_by_slice(position, result):

    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [1., 1., 1.]
    wcs.wcs.crval = [0., 0., 0.]
    wcs.wcs.cdelt = [1e-5, 2e-5, 3e-5]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']

    newwcs = drop_axis_by_slicing(wcs, shape=[10,12,14], dropped_axis=0,
                                  dropped_axis_slice_position=position)

    # drop-by-slicing moves axis to be last
    np.testing.assert_almost_equal(newwcs.wcs.crval[2], result)
    assert all(newwcs.wcs.cdelt == [2e-5,3e-5,1e-5])

def test_drop_by_slice_middle_fullrange():

    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [1., 1., 1.]
    wcs.wcs.crval = [0., 0., 0.]
    wcs.wcs.cdelt = [1e-5, 1e-5, 1e-5]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']

    newwcs = drop_axis_by_slicing(wcs, shape=[10,12,14], dropped_axis=0,
                                  dropped_axis_cdelt='full_range')

    np.testing.assert_almost_equal(newwcs.wcs.crval[2], 5e-5)
    np.testing.assert_almost_equal(newwcs.wcs.cdelt[2], 10e-5)
