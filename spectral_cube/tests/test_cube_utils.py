from radio_beam import Beam

from .test_spectral_cube import cube_and_raw
from ..cube_utils import largest_beam, smallest_beam


def test_largest_beam():

    cube, data = cube_and_raw('522_delta_beams.fits')

    large_beam = largest_beam(cube.beams)

    assert large_beam == cube.beams[2]


def test_smallest_beam():

    cube, data = cube_and_raw('522_delta_beams.fits')

    small_beam = smallest_beam(cube.beams)

    assert small_beam == cube.beams[0]
