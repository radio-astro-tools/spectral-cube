import numpy as np
from numpy.testing import assert_allclose, assert_equal

from astropy.wcs import WCS
from astropy.tests.helper import pytest
from astropy.utils import OrderedDict, NumpyRNGContext

from ..spectral_cube import StokesSpectralCube


class TestStokesSpectralCube():

    def setup_class(self):

        self.wcs = WCS(naxis=4)
        self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ', 'STOKES']
        self.data = np.arange(4)[:,None,None,None] * np.ones((5, 20, 30))

    def test_direct_init(self):
        stokes_data = dict(I=self.data[0], Q=self.data[1], U=self.data[2], V=self.data[3])
        cube = StokesSpectralCube(stokes_data, self.wcs)

    @pytest.mark.parametrize('component', ('I', 'Q', 'U', 'V', 'RR', 'RL', 'LR', 'LL'))
    def test_valid_component_name(self, component):
        stokes_data = {component: self.data[0]}
        cube = StokesSpectralCube(stokes_data, self.wcs)

    @pytest.mark.parametrize('component', ('A', 'B', 'IQUV'))
    def test_invalid_component_name(self, component):
        stokes_data = {component: self.data[0]}
        with pytest.raises(ValueError) as exc:
            cube = StokesSpectralCube(stokes_data, self.wcs)
        assert exc.value.args[0] == "Invalid Stokes component: {0} - should be one of I, Q, U, V, RR, LL, RL, LR".format(component)

    def test_invalid_wcs(self):
        wcs = WCS(naxis=4)  # no celestial, spectral, or stokes axes
        stokes_data = dict(I=self.data[0])
        with pytest.raises(ValueError) as exc:
            cube = StokesSpectralCube(stokes_data, wcs)
        assert exc.value.args[0] == "WCS should contain two celestial axes, one spectral axis, and one Stokes axis"

    def test_attributes(self):
        stokes_data = dict(I=self.data[0], Q=self.data[1], U=self.data[2], V=self.data[3])
        cube = StokesSpectralCube(stokes_data, self.wcs)
        assert_allclose(cube.I.unmasked_data, 0)
        assert_allclose(cube.Q.unmasked_data, 1)
        assert_allclose(cube.U.unmasked_data, 2)
        assert_allclose(cube.V.unmasked_data, 3)

    def test_dir(self):
        stokes_data = dict(I=self.data[0], Q=self.data[1], U=self.data[2])
        cube = StokesSpectralCube(stokes_data, self.wcs)
        for stokes in 'IQU':
            assert stokes in cube.__dir__()
        assert 'V' not in cube.__dir__()

    def test_with_mask(self):

        with NumpyRNGContext(12345):
            mask1 = np.random.random((5, 20, 30)) > 0.2
            mask2 = np.random.random((5, 20, 30)) > 0.4

        stokes_data = dict(I=self.data[0], Q=self.data[1], U=self.data[2], V=self.data[3])
        cube1 = StokesSpectralCube(stokes_data, self.wcs, mask=mask1)

        cube2 = cube1.with_mask(mask2)
        assert_equal(cube2.mask, mask1 & mask2)

    def test_with_mask_invalid_shape(self):

        stokes_data = dict(I=self.data[0], Q=self.data[1], U=self.data[2], V=self.data[3])
        mask1 = np.random.random((5, 20, 15)) > 0.2
        with pytest.raises(ValueError) as exc:
            cube1 = StokesSpectralCube(stokes_data, self.wcs, mask=mask1)
        assert exc.value.args[0] == "Mask shape is not broadcastable to data shape: (5, 20, 15) vs (5, 20, 30)"

    def test_separate_mask(self):

        with NumpyRNGContext(12345):
            mask1 = np.random.random((5, 20, 30)) > 0.2
            mask2 = np.random.random((4, 5, 20, 30)) > 0.4
            mask3 = np.random.random((5, 20, 30))

        stokes_data = dict(I=self.data[0], Q=self.data[1], U=self.data[2], V=self.data[3])
        stokes_mask = dict(I=mask2[0], Q=mask2[1], U=mask2[2], V=mask2[3])
        cube1 = StokesSpectralCube(stokes_data, self.wcs, mask=mask1, stokes_mask=stokes_mask)

        assert_equal(cube1.I.mask, mask1 & mask2[0])
        assert_equal(cube1.Q.mask, mask1 & mask2[1])
        assert_equal(cube1.U.mask, mask1 & mask2[2])
        assert_equal(cube1.V.mask, mask1 & mask2[3])

        cube2 = cube1.I.with_mask(mask3)
        assert_equal(cube2.mask, mask1 & mask2 & mask3)

    def test_separate_mask_invalid_components(self):

        with NumpyRNGContext(12345):
            mask1 = np.random.random((5, 20, 30)) > 0.2
            mask2 = np.random.random((2, 5, 20, 30)) > 0.4

        stokes_data = OrderedDict(I=self.data[0], Q=self.data[1])
        stokes_mask = OrderedDict(RR=mask2[0], LL=mask2[1])

        with pytest.raises(ValueError) as exc:
            cube1 = StokesSpectralCube(stokes_data, self.wcs, mask=mask1, stokes_mask=stokes_mask)
        assert exc.args[0].value == "Stokes mask components (RR, LL) do not match data components (I, Q)"

    def test_separate_mask_invalid_shape(self):

        with NumpyRNGContext(12345):
            mask1 = np.random.random((5, 20, 30)) > 0.2
            mask2 = np.random.random((2, 5, 20, 15)) > 0.4

        stokes_data = OrderedDict(I=self.data[0], Q=self.data[1])
        stokes_mask = OrderedDict(RR=mask2[0], LL=mask2[1])

        with pytest.raises(ValueError) as exc:
            cube1 = StokesSpectralCube(stokes_data, self.wcs, mask=mask1, stokes_mask=stokes_mask)
        assert exc.value.args[0] == "Mask shape for component I is not broadcastable to data shape: (5, 20, 15) vs (5, 20, 30)"
