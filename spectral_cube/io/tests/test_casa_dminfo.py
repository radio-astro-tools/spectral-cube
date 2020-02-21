from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
from pprint import pformat

from ..casa_dminfo import getdminfo

try:
    from casatools import table, image
    CASATOOLS_INSTALLED = True
except ImportError:
    CASATOOLS_INSTALLED = False


SHAPES = [(3,), (5, 3), (8, 4, 2), (4, 8, 3, 1), (133, 400), (100, 211, 201),
          (50, 61, 72, 83), (4, 8, 10, 20, 40)]


@pytest.mark.skipif('not CASATOOLS_INSTALLED')
@pytest.mark.parametrize('shape', SHAPES)
def test_getdminfo(tmp_path, shape):

    filename = str(tmp_path / 'test.image')

    data = np.random.random(shape)

    ia = image()
    ia.fromarray(outfile=filename, pixels=data)
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
