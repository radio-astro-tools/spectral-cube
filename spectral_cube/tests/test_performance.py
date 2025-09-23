"""
Performance-related tests to make sure we don't use more memory than we should.

For now this is just for SpectralCube, not DaskSpectralCube.
"""

import numpy as np

import pytest
import tempfile
import sys

try:
    import tracemalloc
    tracemallocOK = True
except ImportError:
    tracemallocOK = False

from .test_moments import moment_cube
from .helpers import assert_allclose
from ..spectral_cube import SpectralCube
from . import utilities

from astropy import convolution, units as u
from radio_beam import Beam

WINDOWS = sys.platform == "win32"


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


# @pytest.mark.skipif('True')
def test_parallel_performance_smoothing():

    """
    Test parallel performance of spectral_smooth by timing different
    num_cores options with memmap=False and memmap=True.
    """
    import timeit

    setup = 'cube,_ = utilities.generate_gaussian_cube(shape=(300,64,64))'
    stmt = 'result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(20.0), num_cores={0}, use_memmap=False)'

    rslt = {}
    for ncores in (1,2,3,4):
        time = timeit.timeit(stmt=stmt.format(ncores), setup=setup, number=5, globals=globals())
        rslt[ncores] = time

    print()
    print("memmap=False")
    print(rslt)

    setup = 'cube,_ = utilities.generate_gaussian_cube(shape=(300,64,64))'
    stmt = 'result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(20.0), num_cores={0}, use_memmap=True)'

    rslt = {}
    for ncores in (1,2,3,4):
        time = timeit.timeit(stmt=stmt.format(ncores), setup=setup, number=5, globals=globals())
        rslt[ncores] = time

    stmt = 'result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(20.0), num_cores={0}, use_memmap=True, parallel=False)'
    rslt[0] = timeit.timeit(stmt=stmt.format(1), setup=setup, number=5, globals=globals())

    print()
    print("memmap=True")
    print(rslt)


    if False:
        for shape in [(300,64,64), (600,64,64), (900,64,64),
                      (300,128,128), (300,256,256), (900,256,256)]:

            setup = 'cube,_ = utilities.generate_gaussian_cube(shape={0})'.format(shape)
            stmt = 'result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(20.0), num_cores={0}, use_memmap=True)'

            rslt = {}
            for ncores in (1,2,3,4):
                time = timeit.timeit(stmt=stmt.format(ncores), setup=setup, number=5, globals=globals())
                rslt[ncores] = time

            stmt = 'result = cube.spectral_smooth(kernel=convolution.Gaussian1DKernel(20.0), num_cores={0}, use_memmap=True, parallel=False)'
            rslt[0] = timeit.timeit(stmt=stmt.format(1), setup=setup, number=5, globals=globals())

            print()
            print("memmap=True shape={0}".format(shape))
            print(rslt)


# @pytest.mark.skipif('True')
def test_parallel_performance_spatial_smoothing():
    """
    Test parallel performance of spectral_smooth by timing different
    num_cores options with memmap=False and memmap=True.
    """
    import timeit
    from radio_beam import Beam

    setup = 'cube,_ = utilities.generate_gaussian_cube(shape=(300,64,64))'
    stmt = 'result = cube.convolve_to(Beam(cube.beam.major * 1.1), num_cores={0}, use_memmap=False)'

    rslt = {}
    for ncores in (1,2,3,4):
        time = timeit.timeit(stmt=stmt.format(ncores), setup=setup, number=5, globals=globals())
        rslt[ncores] = time

    print()
    print("memmap=False")
    print(rslt)

    setup = 'cube,_ = utilities.generate_gaussian_cube(shape=(300,64,64))'
    stmt = 'result = cube.convolve_to(Beam(cube.beam.major * 1.1), num_cores={0}, use_memmap=True)'

    rslt = {}
    for ncores in (1,2,3,4):
        time = timeit.timeit(stmt=stmt.format(ncores), setup=setup, number=5, globals=globals())
        rslt[ncores] = time

    stmt = 'result = cube.convolve_to(Beam(cube.beam.major * 1.1), num_cores={0}, use_memmap=True, parallel=False)'
    rslt[0] = timeit.timeit(stmt=stmt.format(1), setup=setup, number=5, globals=globals())

    print()
    print("memmap=True")
    print(rslt)


@pytest.mark.skipif('not tracemallocOK or WINDOWS')
def test_memory_usage():
    """
    Make sure that using memmaps happens where expected, for the most part, and
    that memory doesn't get overused.
    """

    ntf = tempfile.NamedTemporaryFile()

    tracemalloc.start()

    snap1 = tracemalloc.take_snapshot()

    # create a 64 MB cube
    cube,_ = utilities.generate_gaussian_cube(shape=[200,200,200])
    sz = _.dtype.itemsize

    snap1b = tracemalloc.take_snapshot()
    diff = snap1b.compare_to(snap1, 'lineno')
    diffvals = np.array([dd.size_diff for dd in diff])
    # at this point, the generated cube should still exist in memory
    assert diffvals.max()*u.B >= 200**3*sz*u.B

    del _
    snap2 = tracemalloc.take_snapshot()
    diff = snap2.compare_to(snap1b, 'lineno')
    assert diff[0].size_diff*u.B < -0.3*u.MB

    cube.write(ntf.name, format='fits')

    # writing the cube should not occupy any more memory
    snap3 = tracemalloc.take_snapshot()

    diff = snap3.compare_to(snap2, 'lineno')
    assert sum([dd.size_diff for dd in diff])*u.B < 100*u.kB

    del cube

    # deleting the cube should remove the 64 MB from memory
    snap4 = tracemalloc.take_snapshot()
    diff = snap4.compare_to(snap3, 'lineno')
    assert diff[0].size_diff*u.B < -200**3*sz*u.B

    cube = SpectralCube.read(ntf.name, format='fits')

    # reading the cube from filename on disk should result in no increase in
    # memory use
    snap5 = tracemalloc.take_snapshot()
    diff = snap5.compare_to(snap4, 'lineno')
    assert diff[0].size_diff*u.B < 1*u.MB

    mask = cube.mask.include()

    snap6 = tracemalloc.take_snapshot()
    diff = snap6.compare_to(snap5, 'lineno')
    assert diff[0].size_diff*u.B >= mask.size*u.B

    filled_data = cube._get_filled_data(use_memmap=True)
    snap7 = tracemalloc.take_snapshot()
    diff = snap7.compare_to(snap6, 'lineno')
    assert diff[0].size_diff*u.B < 100*u.kB

    filled_data = cube._get_filled_data(use_memmap=False)
    snap8 = tracemalloc.take_snapshot()
    diff = snap8.compare_to(snap7, 'lineno')
    assert diff[0].size_diff*u.B > 10*u.MB

    del filled_data

    # cube is <1e8 bytes, so this is use_memmap=False
    filled_data = cube.filled_data[:]
    snap9 = tracemalloc.take_snapshot()
    diff = snap9.compare_to(snap6, 'lineno')
    assert diff[0].size_diff*u.B > 10*u.MB


@pytest.mark.skipif('not tracemallocOK')
def test_memory_usage_coordinates():
    """
    Watch out for high memory usage on huge spatial files
    """

    ntf = tempfile.NamedTemporaryFile()

    tracemalloc.start()

    snap1 = tracemalloc.take_snapshot()

    size = 200

    # create a "flat" cube
    cube,_ = utilities.generate_gaussian_cube(shape=[1,size,size])
    sz = _.dtype.itemsize

    snap1b = tracemalloc.take_snapshot()
    diff = snap1b.compare_to(snap1, 'lineno')
    diffvals = np.array([dd.size_diff for dd in diff])
    # at this point, the generated cube should still exist in memory
    assert diffvals.max()*u.B >= size**2*sz*u.B

    del _
    snap2 = tracemalloc.take_snapshot()
    diff = snap2.compare_to(snap1b, 'lineno')
    assert diff[0].size_diff*u.B < -0.3*u.MB

    print(cube)

    # printing the cube should not occupy any more memory
    # (it will allocate a few bytes for the cache, but should *not*
    # load the full size x size coordinate arrays for RA, Dec
    snap3 = tracemalloc.take_snapshot()
    diff = snap3.compare_to(snap2, 'lineno')
    assert sum([dd.size_diff for dd in diff])*u.B < 100*u.kB
