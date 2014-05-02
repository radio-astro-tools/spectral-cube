from astropy import units as u

from numpy.testing import assert_allclose as assert_allclose_numpy


def assert_allclose(q1, q2):
    """
    Quantity-safe version of Numpy's assert_allclose
    """
    if isinstance(q1, u.Quantity) and isinstance(q2, u.Quantity):
        assert_allclose_numpy(q1.to(q2.unit).value, q2.value)
    elif isinstance(q1, u.Quantity):
        assert_allclose_numpy(q1.value, q2)
    elif isinstance(q2, u.Quantity):
        assert_allclose_numpy(q1, q2.value)
    else:
        assert_allclose_numpy(q1, q2)
