from __future__ import print_function, absolute_import, division

import os
import pytest
import numpy as np
from numpy.testing import assert_equal
from pprint import pformat

from ..casa_dminfo import getdminfo, getdesc
from ...tests.test_casafuncs import make_casa_testimage

try:
    from casatools import table, image
    CASATOOLS_INSTALLED = True
except ImportError:
    CASATOOLS_INSTALLED = False

DATA = os.path.join(os.path.dirname(__file__), 'data')

SHAPES = [(3,), (5, 3), (8, 4, 2), (4, 8, 3, 1), (133, 400), (100, 211, 201),
          (50, 61, 72, 83), (4, 8, 10, 20, 40)]


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
@pytest.mark.parametrize('shape', SHAPES)
def test_getdminfo(tmp_path, shape):

    filename = str(tmp_path / 'test.image')

    data = np.random.random(shape)

    ia = image()
    ia.fromarray(outfile=filename, pixels=data, log=False)
    ia.close()

    tb = table()
    tb.open(filename)
    reference = tb.getdminfo()
    tb.close()

    actual = getdminfo(filename)

    # The easiest way to compare the output is simply to compare the output
    # from pformat (checking for dictionary equality doesn't work because of
    # the Numpy arrays inside).
    assert pformat(actual) == pformat(reference)


def test_getdminfo_large():

    # Check that things continue to work fine when we cross the threshold from
    # a dataset with a size that can be represented by a 32-bit integer to one
    # where the size requires a 64-bit integer. We use pre-generated
    # table.f0 files here since generating these kinds of datasets is otherwise
    # slow and consumes a lot of memory.

    lt32bit = getdminfo(os.path.join(DATA, 'lt32bit.image'))
    assert_equal(lt32bit['*1']['SPEC']['HYPERCUBES']['*1']['CubeShape'], (320, 320, 1, 1920))

    gt32bit = getdminfo(os.path.join(DATA, 'gt32bit.image'))
    assert_equal(gt32bit['*1']['SPEC']['HYPERCUBES']['*1']['CubeShape'], (640, 640, 1, 1920))


@pytest.fixture
def filename(request):
    return request.getfixturevalue(request.param)


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
@pytest.mark.parametrize('filename', ('data_advs', 'data_dvsa', 'data_vsad',
                                      'data_sadv', 'data_sdav', 'data_sdav_beams',
                                      'data_advs_nobeam', 'data_adv',
                                      'data_adv_jybeam_upper', 'data_adv_jybeam_lower',
                                      'data_adv_jybeam_whitespace', 'data_adv_beams',
                                      'data_vad', 'data_vda', 'data_vda_jybeam_upper',
                                      'data_vda_jybeam_lower', 'data_vda_jybeam_whitespace',
                                      'data_vda_beams', 'data_255', 'data_255_delta',
                                      # 'data_455_delta_beams',
                                      'data_522_delta',
                                      # 'data_522_delta_beams'
                                      ),
                         indirect=['filename'])
def test_getdesc(tmp_path, filename):

    casa_filename = str(tmp_path / 'casa.image')

    make_casa_testimage(filename, casa_filename)

    tb = table()
    tb.open(casa_filename)
    desc_reference = tb.getdesc()
    tb.close()

    desc_actual = getdesc(casa_filename)

    # The easiest way to compare the output is simply to compare the output
    # from pformat (checking for dictionary equality doesn't work because of
    # the Numpy arrays inside).
    assert pformat(desc_actual) == pformat(desc_reference)
