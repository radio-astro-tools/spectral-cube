from __future__ import print_function, absolute_import, division

import tempfile

import pytest
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

from numpy.testing import assert_allclose

from ..casa_low_level_io import getdesc
from ..casa_wcs import wcs_casa2astropy
from ...tests.test_casafuncs import make_casa_testimage
from .test_casa_low_level_io import ALL_DATA_FIXTURES
from ...conftest import HEADER_FILENAME

try:
    from casatools import image
    CASATOOLS_INSTALLED = True
except ImportError:
    CASATOOLS_INSTALLED = False


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


def assert_header_correct(casa_filename):

    fits_filename = tempfile.mktemp()

    # Use CASA to convert back to FITS and use that header as the reference

    ia = image()
    ia.open(casa_filename)
    ia.tofits(fits_filename, stokeslast=False)
    ia.done()
    ia.close()

    # Parse header with WCS - for the purposes of this function
    # we are not interested in keywords/values not in WCS
    reference_wcs = WCS(fits_filename)
    reference_header = reference_wcs.to_header()

    # Now use our wcs_casa2astropy function to create the header and compare
    # the results.
    desc = getdesc(casa_filename)
    actual_wcs = wcs_casa2astropy(desc['_keywords_']['coords'])
    actual_header = actual_wcs.to_header()

    assert sorted(actual_header) == sorted(reference_header)

    for key in reference_header:
        if isinstance(actual_header[key], str):
            assert actual_header[key] == reference_header[key]
        else:
            assert_allclose(actual_header[key], reference_header[key])


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
@pytest.mark.parametrize('filename', ALL_DATA_FIXTURES, indirect=['filename'])
def test_wcs_casa2astropy(tmp_path, filename):
    casa_filename = str(tmp_path / 'casa.image')
    make_casa_testimage(filename, casa_filename)
    assert_header_correct(casa_filename)


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
def test_wcs_casa2astropy_linear(tmp_path):

    # Test that things work properly when the WCS coordinates aren't set

    casa_filename = str(tmp_path / 'test.image')

    data = np.random.random((3, 4, 5, 6, 7))

    ia = image()
    ia.fromarray(outfile=casa_filename, pixels=data, log=False)
    ia.close()

    assert_header_correct(casa_filename)


def header_copy_with(**kwargs):
    header = fits.Header.fromtextfile(HEADER_FILENAME).copy()
    header.update(kwargs)
    return header


ALL_HEADERS = [
    header_copy_with(CTYPE1='GLON-TAN', CTYPE2='GLAT-TAN'),
    header_copy_with(CTYPE1='SLON-TAN', CTYPE2='SLAT-TAN'),
    header_copy_with(CTYPE1='ELON-TAN', CTYPE2='ELAT-TAN'),
    header_copy_with(CTYPE1='HLON-TAN', CTYPE2='HLAT-TAN'),
    header_copy_with(SPECSYS='TOPOCENT'),
    header_copy_with(SPECSYS='GEOCENTR'),
    header_copy_with(SPECSYS='BARYCENT'),
    header_copy_with(SPECSYS='HELIOCEN'),
    header_copy_with(SPECSYS='LSRK'),
    header_copy_with(SPECSYS='LSRD'),
    header_copy_with(SPECSYS='GALACTOC'),
    header_copy_with(SPECSYS='LOCALGRP'),
    header_copy_with(SPECSYS='CMBDIPOL'),
    header_copy_with(SPECSYS='SOURCE'),
    header_copy_with(RADESYS='FK4'),
    header_copy_with(RADESYS='FK4-NO-E'),
    header_copy_with(RADESYS='FK5'),
    header_copy_with(RADESYS='ICRS'),
    header_copy_with(EQUINOX=1950.),
    header_copy_with(EQUINOX=1979.9),
    header_copy_with(EQUINOX=2000),
    header_copy_with(EQUINOX=2010),
    header_copy_with(CTYPE3='FREQ', CUNIT3='GHz', CRVAL3=100., CDELT3=1.),
    header_copy_with(CTYPE3='WAVE', CUNIT3='m', CRVAL3=1e-6, CDELT3=1e-8),
    header_copy_with(CTYPE3='VOPT'),
    header_copy_with(CTYPE3='VRAD')
]

@pytest.mark.skipif('not CASATOOLS_INSTALLED')
@pytest.mark.parametrize('header', ALL_HEADERS)
def test_wcs_casa2astropy_additional(tmp_path, header):

    # More cases to improve coverage

    casa_filename = str(tmp_path / 'casa.image')
    fits_filename = str(tmp_path / 'casa.fits')

    fits.writeto(fits_filename, np.ones((2, 3, 4, 5)), header)

    make_casa_testimage(fits_filename, casa_filename)

    assert_header_correct(casa_filename)
