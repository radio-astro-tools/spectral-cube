from __future__ import print_function, absolute_import, division

from astropy import units as u

from numpy.testing import assert_allclose as assert_allclose_numpy, assert_array_equal


def assert_allclose(q1, q2, **kwargs):
    """
    Quantity-safe version of Numpy's assert_allclose
    """
    if isinstance(q1, u.Quantity) and isinstance(q2, u.Quantity):
        assert_allclose_numpy(q1.to(q2.unit).value, q2.value, **kwargs)
    elif isinstance(q1, u.Quantity):
        assert_allclose_numpy(q1.value, q2, **kwargs)
    elif isinstance(q2, u.Quantity):
        assert_allclose_numpy(q1, q2.value, **kwargs)
    else:
        assert_allclose_numpy(q1, q2, **kwargs)
