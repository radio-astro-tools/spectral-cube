import numpy as np
from radio_beam import Beam
from astropy.io import fits
from astropy import units as u

from .test_spectral_cube import cube_and_raw, path
from ..cube_utils import largest_beam, smallest_beam, beams_to_bintable


def test_largest_beam(data_522_delta_beams, use_dask):

    cube, data = cube_and_raw(data_522_delta_beams, use_dask=use_dask)

    large_beam = largest_beam(cube.beams)

    assert large_beam == cube.beams[2]


def test_smallest_beam(data_522_delta_beams, use_dask):

    cube, data = cube_and_raw(data_522_delta_beams, use_dask=use_dask)

    small_beam = smallest_beam(cube.beams)

    assert small_beam == cube.beams[0]


def test_beams_to_bintable_cube(data_522_delta_beams, use_dask):

    cube, data = cube_and_raw(data_522_delta_beams, use_dask=use_dask)

    hdul = fits.open(data_522_delta_beams)
    beamtable = hdul[1]

    bms = beams_to_bintable(cube.beams)

    assert np.all(beamtable.data == bms.data)

    assert bms.header['NPOL'] == 1
    assert bms.header['NCHAN'] == 5

    hdul.close()


def test_beams_to_bintable():
    """ Check that NPOL is set """
    beamlist = [Beam(1*u.arcsec)]*2
    beamhdu = beams_to_bintable(beamlist)

    assert beamhdu.header['NPOL'] == 0
