import pytest
import warnings

from astropy.io import fits
import numpy as np

from astropy.utils.exceptions import AstropyDeprecationWarning

from ..wcs_utils import (WCS, drop_axis, wcs_swapaxes, add_stokes_axis_to_wcs,
                         axis_names, reindex_wcs, slice_wcs, check_equality,
                         strip_wcs_from_header)
from ..utils import WCSWarning
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


def test_deprecated_wrappers():
    # These are thin wrappers around functionality in astropy, kept so that
    # existing imports keep working, so we only check that they warn and that
    # the behavior specific to the wrappers is preserved.
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VRAD']
    wcs.pixel_shape = (4, 5, 6)

    with pytest.warns(AstropyDeprecationWarning, match='WCS.swapaxes'):
        swapped = wcs_swapaxes(wcs, 0, 2)
    assert list(swapped.wcs.ctype) == ['VRAD', 'DEC--TAN', 'RA---TAN']

    with pytest.warns(AstropyDeprecationWarning, match='axis_type_names'):
        assert axis_names(wcs) == ['RA', 'DEC', 'VRAD']

    # The Stokes axis is referenced to the first pixel and has a length of 1
    with pytest.warns(AstropyDeprecationWarning,
                      match='astropy.wcs.utils.add_stokes_axis_to_wcs'):
        stokes = add_stokes_axis_to_wcs(wcs, 0)
    assert stokes.wcs.crpix[0] == 1
    assert stokes.wcs.crval[0] == 1
    assert stokes.pixel_shape == (1, 4, 5, 6)

    # Keeping only one of the celestial axes removes the projection
    with pytest.warns(WCSWarning, match='celestial'):
        with pytest.warns(AstropyDeprecationWarning, match='WCS.sub'):
            reindexed = reindex_wcs(wcs, np.array([2, 0]))
    assert list(reindexed.wcs.ctype) == ['VRAD', 'RA']
    assert reindexed.pixel_shape == (6, 4)

    with pytest.warns(AstropyDeprecationWarning):
        with pytest.raises(TypeError, match='must be an ndarray'):
            reindex_wcs(wcs, [0, 1])


def test_pixel_shape_preserved_on_reindex():
    # Regression test for #1000: drop_axis used to silently reset pixel_shape
    # (NAXISj) to zeros, so e.g. a moment's WCS reported NAXIS: 0 0 instead of
    # the cube's spatial shape.
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR', 'VRAD']
    wcs.pixel_shape = (4321, 2161, 118)

    dropped = drop_axis(wcs, 2)
    assert dropped.pixel_shape == (4321, 2161)

    dropped = drop_axis(wcs, 0)
    assert dropped.pixel_shape == (2161, 118)

    # If the source has no pixel_shape, the result shouldn't either.
    bare = WCS(naxis=3)
    assert drop_axis(bare, 2).pixel_shape is None


def test_drop_axis_preserves_pv():
    # Projection parameters on the celestial axes must survive dropping the
    # spectral axis, otherwise e.g. moment maps of NCP-like SIN cubes get
    # incorrect celestial coordinates.
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'VRAD']
    wcs.wcs.crval = [10., 60., 0.]
    wcs.wcs.cdelt = [-0.1, 0.1, 1.]
    wcs.wcs.set_pv([(2, 1, 0.), (2, 2, 1 / np.tan(np.radians(60)))])
    wcs.wcs.set()

    dropped = drop_axis(wcs, 2)
    assert dropped.wcs.get_pv() == wcs.wcs.get_pv()
    np.testing.assert_allclose(dropped.pixel_to_world_values(3, 4),
                               wcs.pixel_to_world_values(3, 4, 0)[:2])


@pytest.mark.parametrize('matrix', ['pc', 'cd'])
@pytest.mark.parametrize('dropax', [0, 1])
def test_drop_celestial_axis_rotated(matrix, dropax):
    # Dropping a single celestial axis strips the projection with a warning,
    # and the coupling between the dropped and the remaining axes is ignored.
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VRAD']
    wcs.wcs.crval = [10., 20., 5.]
    rotated = np.array([[-0.9, 0.3, 0.], [0.3, 0.9, 0.], [0., 0., 0.5]])
    if matrix == 'pc':
        wcs.wcs.pc = rotated
    else:
        wcs.wcs.cd = rotated
    wcs.wcs.set()

    with pytest.warns(WCSWarning, match='celestial projection'):
        dropped = drop_axis(wcs, dropax)

    keep = [ii for ii in range(3) if ii != dropax]
    assert list(dropped.wcs.ctype) == [wcs.wcs.ctype[ii].split('-')[0]
                                       for ii in keep]
    pixel_scale = wcs.pixel_scale_matrix[np.ix_(keep, keep)]
    np.testing.assert_allclose(dropped.pixel_scale_matrix, pixel_scale)


def test_drop_axis_negative_index():
    # Negative indices count from the last axis, including when the dropped
    # axis is coupled to the remaining ones or is a single celestial axis.
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['LINEAR', 'FREQ', 'STOKES']
    wcs.wcs.pc = [[1., 0.2, 0.], [0.2, 1., 0.3], [0., 0.3, 1.]]
    wcs.wcs.set()

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        dropped = drop_axis(wcs, -1)

    assert list(dropped.wcs.ctype) == ['LINEAR', 'FREQ']
    np.testing.assert_allclose(dropped.pixel_scale_matrix,
                               drop_axis(wcs, 2).pixel_scale_matrix)

    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'VRAD']
    wcs.wcs.set()

    with pytest.warns(WCSWarning, match='celestial projection'):
        dropped = drop_axis(wcs, -3)

    assert list(dropped.wcs.ctype) == ['DEC', 'VRAD']


def test_wcs_slice():
    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [50., 45., 30.]
    wcs_new = slice_wcs(wcs, (slice(10,20), slice(None), slice(20,30)))
    np.testing.assert_allclose(wcs_new.wcs.crpix, [30., 45., 20.])


@pytest.mark.parametrize('start', [-2, -10, -20])
def test_wcs_slice_negative_start(start):
    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ['FREQ']
    wcs.wcs.crpix = [2.5]
    wcs.wcs.crval = [115e9]
    wcs.wcs.cdelt = [1e6]
    pixels = np.arange(10)
    expected = wcs.wcs_pix2world(pixels, 0)[0][start:]
    sliced = slice_wcs(wcs, slice(start, None), shape=(10,))
    actual = sliced.wcs_pix2world(np.arange(len(expected)), 0)[0]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('start', [-2, -40])
def test_wcs_slice_negative_start_drop_axes(axis, start):
    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [2.5, 3.5, 4.5]
    wcs.wcs.crval = [100., 200., 300.]
    wcs.wcs.cdelt = [1., 2., 3.]
    shape = (10, 20, 30)
    view = [0, 0, 0]
    view[axis] = slice(start, None)
    pixels = np.arange(shape[axis])[start:]
    coordinates = np.zeros((len(pixels), 3))
    coordinates[:, 2-axis] = pixels
    expected = wcs.wcs_pix2world(coordinates, 0)[:, 2-axis]
    sliced = slice_wcs(wcs, view, shape=shape, drop_degenerate=True)
    actual = sliced.wcs_pix2world(np.arange(len(pixels)), 0)[0]
    np.testing.assert_allclose(actual, expected)

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

def test_reversal_roundtrip():
    wcs = WCS(naxis=3)
    wcs.wcs.crpix = [50., 45., 30.]
    wcs.wcs.crval = [0., 0., 0.]
    wcs.wcs.cdelt = [1., 1., 1.]
    wcs_new = slice_wcs(wcs, (slice(None, None, -1), slice(None), slice(None)),
                        shape=[100., 150., 200.])
    spaxis = wcs.sub([0]).wcs_pix2world(np.arange(100), 0)

    new_spaxis = wcs_new.sub([0]).wcs_pix2world(np.arange(100), 0)

    np.testing.assert_allclose(spaxis, new_spaxis[::-1])

    re_reverse = slice_wcs(wcs_new, (slice(None, None, -1), slice(None), slice(None)),
                           shape=[100., 150., 200.])
    new_spaxis = re_reverse.sub([0]).wcs_pix2world(np.arange(100), 0)

    np.testing.assert_allclose(spaxis, new_spaxis[::-1])

    #These are NOT equal, but they are equivalent: CRVAL and CRPIX are shifted
    #by an acceptable amount
    # assert check_equality(wcs, re_reverse)

    re_re_reverse = slice_wcs(re_reverse, (slice(None, None, -1), slice(None),
                                           slice(None)),
                              shape=[100., 150., 200.])
    re_re_re_reverse = slice_wcs(re_re_reverse, (slice(None, None, -1),
                                                 slice(None), slice(None)),
                                 shape=[100., 150., 200.])

    assert check_equality(re_re_re_reverse, re_reverse)

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

@pytest.mark.parametrize('fn', ('cubewcs1.hdr', 'cubewcs2.hdr'))
def test_strip_wcs(fn):

    header1 = fits.Header.fromtextfile(path(fn))
    header1_stripped = strip_wcs_from_header(header1)

    with open(path(fn),'r') as fh:
        hdrlines = fh.readlines()

    newfn = fn.replace('.hdr', '_blanks.hdr')

    hdrlines.insert(-20,"\n")
    hdrlines.insert(-1,"\n")
    with open(path(newfn),'w') as fh:
        fh.writelines(hdrlines)

    header2 = fits.Header.fromtextfile(path(newfn))
    header2_stripped = strip_wcs_from_header(header2)

    assert header1_stripped == header2_stripped

def test_wcs_slice_unmatched_celestial():
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']
    wcs.wcs.crpix = [50., 45., 30.]

    # drop RA
    with warnings.catch_warnings(record=True) as wrn:
        wcs_new = drop_axis(wcs, 0)

    assert 'is being removed' in str(wrn[-1].message)

    # drop Dec
    with warnings.catch_warnings(record=True) as wrn:
        wcs_new = drop_axis(wcs, 1)

    assert 'is being removed' in str(wrn[-1].message)

    with warnings.catch_warnings(record=True) as wrn:
        wcs_new = slice_wcs(wcs, (slice(10,20), 0, slice(20,30)),
                            drop_degenerate=True)

    assert 'is being removed' in str(wrn[-1].message)

def test_wcs_downsampling():
    """
    Regression tests for #525

    These are a series of simple tests I verified with pen and paper, but it's
    always worth checking me again.
    """
    wcs = WCS(naxis=1)
    wcs.wcs.ctype = ['FREQ',]
    wcs.wcs.crpix = [1.,]

    nwcs = slice_wcs(wcs, slice(0, None, 1))
    assert nwcs.wcs.crpix[0] == 1

    nwcs = slice_wcs(wcs, slice(0, None, 2))
    assert nwcs.wcs.crpix[0] == 0.75

    nwcs = slice_wcs(wcs, slice(0, None, 4))
    assert nwcs.wcs.crpix[0] == 0.625

    nwcs = slice_wcs(wcs, slice(2, None, 1))
    assert nwcs.wcs.crpix[0] == -1

    nwcs = slice_wcs(wcs, slice(2, None, 2))
    assert nwcs.wcs.crpix[0] == -0.25

    nwcs = slice_wcs(wcs, slice(2, None, 4))
    assert nwcs.wcs.crpix[0] == 0.125
