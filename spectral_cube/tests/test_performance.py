"""
Performance-related tests to make sure we don't use more memory than we should
"""

from __future__ import print_function, absolute_import, division

from .test_moments import moment_cube
from .helpers import assert_allclose
from ..spectral_cube import SpectralCube

def find_base_nbytes(obj):
    # from http://stackoverflow.com/questions/34637875/size-of-numpy-strided-array-broadcast-array-in-memory
    if obj.base is not None:
        return find_base_nbytes(obj.base)
    return obj.nbytes

def test_pix_size():
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)

    s,y,x = sc._pix_size()

    # float64 by default
    bytes_per_pix = 8

    assert find_base_nbytes(s) == sc.shape[0]*bytes_per_pix
    assert find_base_nbytes(y) == sc.shape[1]*sc.shape[2]*bytes_per_pix
    assert find_base_nbytes(x) == sc.shape[1]*sc.shape[2]*bytes_per_pix

def test_compare_pix_size_approaches():
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)

    sa,ya,xa = sc._pix_size()
    s,y,x = (sc._pix_size_slice(ii) for ii in range(3))

    assert_allclose(sa, s)
    assert_allclose(ya, y)
    assert_allclose(xa, x)

def test_pix_cen():
    mc_hdu = moment_cube()
    sc = SpectralCube.read(mc_hdu)

    s,y,x = sc._pix_cen()

    # float64 by default
    bytes_per_pix = 8

    assert find_base_nbytes(s) == sc.shape[0]*bytes_per_pix
    assert find_base_nbytes(y) == sc.shape[1]*sc.shape[2]*bytes_per_pix
    assert find_base_nbytes(x) == sc.shape[1]*sc.shape[2]*bytes_per_pix
