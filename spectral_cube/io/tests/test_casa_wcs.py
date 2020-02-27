from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
from astropy.wcs import WCS

from numpy.testing import assert_allclose

from ..casa_low_level_io import getdesc
from ..casa_wcs import wcs_casa2astropy
from ...tests.test_casafuncs import make_casa_testimage
from .test_casa_low_level_io import ALL_DATA_FIXTURES

try:
    from casatools import image
    CASATOOLS_INSTALLED = True
except ImportError:
    CASATOOLS_INSTALLED = False


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
@pytest.mark.parametrize('filename', ALL_DATA_FIXTURES, indirect=['filename'])
def test_wcs_casa2astropy(tmp_path, filename):

    casa_filename = str(tmp_path / 'casa.image')
    fits_filename = str(tmp_path / 'casa.fits')
    make_casa_testimage(filename, casa_filename)

    # Use CASA to convert back to FITS and use that header as the reference

    ia = image()
    ia.open(casa_filename)
    ia.tofits(fits_filename, stokeslast=False)
    ia.done()

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
            if 'CDELT' in key:
                # FIXME:For some reason the CDELT for spectral axes disagree
                assert_allclose(actual_header[key], reference_header[key], rtol=1.e-4)
            else:
                assert_allclose(actual_header[key], reference_header[key])


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
def test_wcs_casa2astropy_linear(tmp_path):

    # Test that things work properly when the WCS coordinates aren't set

    casa_filename = str(tmp_path / 'test.image')
    fits_filename = str(tmp_path / 'test.fits')

    data = np.random.random((3, 4, 5, 6, 7))

    ia = image()
    ia.fromarray(outfile=casa_filename, pixels=data, log=False)
    ia.close()

    # Use CASA to convert back to FITS and use that header as the reference

    ia = image()
    ia.open(casa_filename)
    ia.tofits(fits_filename, stokeslast=False)
    ia.done()

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
        print(key)
        if isinstance(actual_header[key], str):
            assert actual_header[key] == reference_header[key]
        else:
            assert_allclose(actual_header[key], reference_header[key])
