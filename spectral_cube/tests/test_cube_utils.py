import pytest

from .test_spectral_cube import cube_and_raw
from ..cube_utils import largest_beam, smallest_beam

try:
    from radio_beam import beam, Beam
    RADIO_BEAM_INSTALLED = True
except ImportError:
    RADIO_BEAM_INSTALLED = False

@pytest.mark.skipif('not RADIO_BEAM_INSTALLED')
def test_largest_beam():

    cube, data = cube_and_raw('522_delta_beams.fits')

    large_beam = largest_beam(cube.beams)

    assert large_beam == cube.beams[2]


@pytest.mark.skipif('not RADIO_BEAM_INSTALLED')
def test_smallest_beam():

    cube, data = cube_and_raw('522_delta_beams.fits')

    small_beam = smallest_beam(cube.beams)

    assert small_beam == cube.beams[0]
