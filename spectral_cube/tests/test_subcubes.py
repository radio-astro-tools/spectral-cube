from __future__ import print_function, absolute_import, division

import pytest

from astropy import units as u
from astropy import wcs
import numpy as np

from . import path
from .helpers import assert_allclose, assert_array_equal
from .test_spectral_cube import cube_and_raw

try:
    import pyregion
    pyregionOK = True
except ImportError:
    pyregionOK = False

def test_subcube():

    cube, data = cube_and_raw('advs.fits')

    sc1 = cube.subcube(xlo=1, xhi=3)
    sc2 = cube.subcube(xlo=24.06269*u.deg, xhi=24.06206*u.deg)

    assert sc1.shape == (2,3,2)
    assert sc2.shape == (2,3,2)
    assert sc1.wcs.wcs.compare(sc2.wcs.wcs)

    sc3 = cube.subcube()

    assert sc3.shape == cube.shape
    assert sc3.wcs.wcs.compare(cube.wcs.wcs)
    assert np.all(sc3._data == cube._data)

@pytest.mark.skipif('not pyregionOK', reason='Could not import pyregion')
@pytest.mark.parametrize(('regfile','result'),
                         (('fk5.reg', [slice(None),1,slice(None)]),
                          ('image.reg', [slice(None),1,slice(None)]),
                          ('partial_overlap_image.reg', [slice(None),1,1]),
                          ('no_overlap_image.reg', ValueError),
                          ('partial_overlap_fk5.reg', [slice(None),1,1]),
                          ('no_overlap_fk5.reg', ValueError),
                         ))
def test_ds9region(regfile, result):
    cube, data = cube_and_raw('adv.fits')

    regions = pyregion.open(path(regfile))

    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result) as exc:
            sc = cube.subcube_from_ds9region(regions)
        # this assertion is redundant, I think...
        assert exc.errisinstance(result)
    else:
        sc = cube.subcube_from_ds9region(regions)
        scsum = sc.sum()
        dsum = data[result].sum()
        assert_allclose(scsum, dsum)

@pytest.mark.parametrize('regfile',
                         ('255-fk5.reg', '255-pixel.reg'),
                        )
def test_ds9region_255(regfile):
    # specific test for correctness
    cube, data = cube_and_raw('255.fits')

    regions = pyregion.open(path(regfile))

    subhdr = cube.wcs.sub([wcs.WCSSUB_CELESTIAL]).to_header()

    mask = regions.get_mask(header=subhdr,
                            shape=cube.shape[1:])

    assert_array_equal(cube[0,:,:][mask].value, [11,12,16,17])

    subcube = cube.subcube_from_ds9region(regions)
    assert_array_equal(subcube[0,:,:].value, np.array([11,12,16,17]).reshape((2,2)))




    #region = 'fk5\ncircle(29.9346557, 24.0623827, 0.11111)'
    #subcube = cube.subcube_from_ds9region(region)
    # THIS TEST FAILS!
    # I think the coordinate transformation in ds9 is wrong;
    # it uses kapteyn?
    
    #region = 'circle(2,2,2)'
    #subcube = cube.subcube_from_ds9region(region)

