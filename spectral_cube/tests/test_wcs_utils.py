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
