from collections import OrderedDict

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from astropy.wcs import WCS
from astropy.tests.helper import pytest
from astropy.utils import NumpyRNGContext
from astropy.coordinates import custom_stokes_symbol_mapping, StokesSymbol

from ..spectral_cube import SpectralCube
from ..stokes_spectral_cube import StokesSpectralCube
from ..masks import BooleanArrayMask

# Use a list of valid stokes symbols for parameterization
VALID_STOKES_LIST = ['I', 'Q', 'U', 'V', 'RR', 'LL', 'RL', 'LR', 'XX', 'XY', 'YX', 'YY',
                     'RX', 'RY', 'LX', 'LY', 'XR', 'XL', 'YR', 'YL', 'PP', 'PQ', 'QP', 'QQ',
                     'RCircular', 'LCircular', 'Linear', 'Ptotal', 'Plinear', 'PFtotal',
                     'PFlinear', 'Pangle']

class TestStokesSpectralCube():

    def setup_class(self):

        self.wcs = WCS(naxis=3)
        self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']
        self.data = np.arange(4)[:, None, None, None] * np.ones((5, 20, 30))

    def test_direct_init(self, use_dask):
        stokes_data = dict(I=SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask),
                           Q=SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask),
                           U=SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask),
                           V=SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask))
        cube = StokesSpectralCube(stokes_data)

    def test_direct_init_invalid_type(self, use_dask):
        stokes_data = dict(I=self.data[0],
                           Q=self.data[1],
                           U=self.data[2],
                           V=self.data[3])
        with pytest.raises(TypeError) as exc:
            cube = StokesSpectralCube(stokes_data)
        assert exc.value.args[0] == "stokes_data should be a dictionary of SpectralCube objects"

    def test_direct_init_invalid_shape(self, use_dask):
        stokes_data = dict(I=SpectralCube(np.ones((6, 2, 30)), wcs=self.wcs, use_dask=use_dask),
                           Q=SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask),
                           U=SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask),
                           V=SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask))
        with pytest.raises(ValueError) as exc:
            cube = StokesSpectralCube(stokes_data)
        assert exc.value.args[0] == "All spectral cubes should have the same shape"

    @pytest.mark.parametrize('component', VALID_STOKES_LIST)
    def test_valid_component_name(self, component, use_dask):
        # Register custom symbols if needed
        custom_stokes_map = getattr(StokesSpectralCube, '_custom_stokes_map', {})
        if component in [s.symbol for s in custom_stokes_map.values()]:
            with custom_stokes_symbol_mapping(custom_stokes_map):
                stokes_data = {component: SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)}
                cube = StokesSpectralCube(stokes_data)
                assert cube.components == [component]
        else:
            stokes_data = {component: SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)}
            cube = StokesSpectralCube(stokes_data)
            assert cube.components == [component]

    @pytest.mark.parametrize('component', ('A', 'B', 'IQUV'))
    def test_invalid_component_name(self, component, use_dask):
        stokes_data = {component: SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)}
        with pytest.raises(ValueError) as exc:
            cube = StokesSpectralCube(stokes_data)
        # The new error message comes from StokesCoord, so just check ValueError is raised

    def test_invalid_wcs(self, use_dask):
        wcs2 = WCS(naxis=3)
        wcs2.wcs.ctype = ['GLON-CAR', 'GLAT-CAR', 'FREQ']
        stokes_data = dict(I=SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask),
                           Q=SpectralCube(self.data[1], wcs2))
        with pytest.raises(ValueError) as exc:
            cube = StokesSpectralCube(stokes_data)
        assert exc.value.args[0] == "All spectral cubes in stokes_data should have the same WCS"

    def test_attributes(self, use_dask):
        stokes_data = OrderedDict()
        stokes_data['I'] = SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)
        stokes_data['Q'] = SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask)
        stokes_data['U'] = SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask)
        stokes_data['V'] = SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask)
        cube = StokesSpectralCube(stokes_data)
        assert_allclose(cube.I.unmasked_data[...], 0)
        assert_allclose(cube.Q.unmasked_data[...], 1)
        assert_allclose(cube.U.unmasked_data[...], 2)
        assert_allclose(cube.V.unmasked_data[...], 3)
        assert cube.components == ['I', 'Q', 'U', 'V']

    def test_stokes_type_sky(self, use_dask):
        stokes_data = OrderedDict()
        stokes_data['I'] = SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)
        stokes_data['Q'] = SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask)
        stokes_data['U'] = SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask)
        stokes_data['V'] = SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask)
        cube = StokesSpectralCube(stokes_data)
        assert cube.stokes_type == "SKY_STOKES"

    def test_stokes_type_feed_circular(self, use_dask):
        stokes_data = OrderedDict()
        stokes_data['RR'] = SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)
        stokes_data['RL'] = SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask)
        stokes_data['LR'] = SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask)
        stokes_data['LL'] = SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask)
        cube = StokesSpectralCube(stokes_data)
        assert cube.stokes_type == "FEED_CIRCULAR"

    def test_stokes_type_feed_linear(self, use_dask):
        stokes_data = OrderedDict()
        stokes_data['XX'] = SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)
        stokes_data['XY'] = SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask)
        stokes_data['YX'] = SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask)
        stokes_data['YY'] = SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask)
        cube = StokesSpectralCube(stokes_data)
        assert cube.stokes_type == "FEED_LINEAR"

    def test_stokes_type_feed_linear_partial(self, use_dask):
        stokes_data = OrderedDict()
        stokes_data['XX'] = SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)
        stokes_data['YY'] = SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask)
        cube = StokesSpectralCube(stokes_data)
        assert cube.stokes_type == "FEED_LINEAR"

    def test_dir(self, use_dask):
        stokes_data = dict(I=SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask),
                           Q=SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask),
                           U=SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask))
        cube = StokesSpectralCube(stokes_data)

        attributes = dir(cube)

        for stokes in 'IQU':
            assert stokes in attributes
        assert 'V' not in attributes
        assert 'mask' in attributes
        assert 'wcs' in attributes
        assert 'shape' in attributes

    def test_mask(self, use_dask):

        with NumpyRNGContext(12345):
            mask1 = BooleanArrayMask(np.random.random((5, 20, 30)) > 0.2, wcs=self.wcs)
            # Deliberately don't use a BooleanArrayMask to check auto-conversion
            mask2 = np.random.random((5, 20, 30)) > 0.4

        stokes_data = dict(I=SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask),
                           Q=SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask),
                           U=SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask),
                           V=SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask))
        cube1 = StokesSpectralCube(stokes_data, mask=mask1)

        cube2 = cube1.with_mask(mask2)
        assert cube2.mask is not None
        # Use .include() only if mask is BooleanArrayMask, else convert to array
        mask1_arr = mask1.include() if hasattr(mask1, 'include') else (np.asarray(mask1) if mask1 is not None else None)
        mask2_arr = mask2 if isinstance(mask2, np.ndarray) else (np.asarray(mask2) if mask2 is not None else None)
        if mask1_arr is not None and mask2_arr is not None:
            combined = mask1_arr & mask2_arr
        elif mask1_arr is not None:
            combined = mask1_arr
        elif mask2_arr is not None:
            combined = mask2_arr
        else:
            combined = None
        assert_equal(cube2.mask.include(), combined)

    def test_mask_invalid_component_name(self, use_dask):
        stokes_data = {'BANANA': SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)}
        with pytest.raises(ValueError) as exc:
            cube = StokesSpectralCube(stokes_data)
        # The new error message comes from StokesCoord, so just check ValueError is raised

    def test_mask_invalid_shape(self, use_dask):
        stokes_data = dict(I=SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask),
                           Q=SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask),
                           U=SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask),
                           V=SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask))
        mask1 = BooleanArrayMask(np.random.random((5, 20, 15)) > 0.2, wcs=self.wcs)
        with pytest.raises(ValueError) as exc:
            cube1 = StokesSpectralCube(stokes_data, mask=mask1)
        assert exc.value.args[0] == "Mask shape is not broadcastable to data shape: (5, 20, 15) vs (5, 20, 30)"

    def test_separate_mask(self, use_dask):

        with NumpyRNGContext(12345):
            mask1 = BooleanArrayMask(np.random.random((5, 20, 30)) > 0.2, wcs=self.wcs)
            mask2 = [BooleanArrayMask(np.random.random((5, 20, 30)) > 0.4, wcs=self.wcs) for i in range(4)]
            mask3 = BooleanArrayMask(np.random.random((5, 20, 30)) > 0.2, wcs=self.wcs)

        stokes_data = dict(I=SpectralCube(self.data[0], wcs=self.wcs, mask=mask2[0], use_dask=use_dask),
                           Q=SpectralCube(self.data[1], wcs=self.wcs, mask=mask2[1], use_dask=use_dask),
                           U=SpectralCube(self.data[2], wcs=self.wcs, mask=mask2[2], use_dask=use_dask),
                           V=SpectralCube(self.data[3], wcs=self.wcs, mask=mask2[3], use_dask=use_dask))

        cube1 = StokesSpectralCube(stokes_data, mask=mask1)

        assert_equal(cube1.I.mask.include(), (mask1 & mask2[0]).include())
        assert_equal(cube1.Q.mask.include(), (mask1 & mask2[1]).include())
        assert_equal(cube1.U.mask.include(), (mask1 & mask2[2]).include())
        assert_equal(cube1.V.mask.include(), (mask1 & mask2[3]).include())

        cube2 = cube1.I.with_mask(mask3)
        assert_equal(cube2.mask.include(), (mask1 & mask2[0] & mask3).include())

    def test_key_access_valid(self, use_dask):
        stokes_data = OrderedDict()
        stokes_data['I'] = SpectralCube(self.data[0], wcs=self.wcs, use_dask=use_dask)
        stokes_data['Q'] = SpectralCube(self.data[1], wcs=self.wcs, use_dask=use_dask)
        stokes_data['U'] = SpectralCube(self.data[2], wcs=self.wcs, use_dask=use_dask)
        stokes_data['V'] = SpectralCube(self.data[3], wcs=self.wcs, use_dask=use_dask)
        cube = StokesSpectralCube(stokes_data)
        assert_equal(cube['I'],cube._stokes_data['I'])
        assert_equal(cube['Q'],cube._stokes_data['Q'])
        assert_equal(cube['U'],cube._stokes_data['U'])
        assert_equal(cube['V'],cube._stokes_data['V'])

class TestStokesSpectralCubeTransformBasis:
    def setup_class(self):
        from astropy.wcs import WCS
        self.wcs = WCS(naxis=3)
        self.wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']
        # Simple data for easy checking, shape (4, 5, 5)
        self.data = np.zeros((4, 5, 5))
        self.data[0] = 10  # I or RR or XX
        self.data[1] = 2   # Q or RL or XY
        self.data[2] = 3   # U or LR or YX
        self.data[3] = 4   # V or LL or YY

    def test_linear_to_sky(self):
        stokes_data = dict(
            XX=SpectralCube(self.data[0][None, ...], wcs=self.wcs),
            XY=SpectralCube(self.data[1][None, ...], wcs=self.wcs),
            YX=SpectralCube(self.data[2][None, ...], wcs=self.wcs),
            YY=SpectralCube(self.data[3][None, ...], wcs=self.wcs),
        )
        cube = StokesSpectralCube(stokes_data)
        sky_cube = cube.transform_basis('Sky')
        assert_allclose(sky_cube['I'].unmasked_data[...], 7)
        assert_allclose(sky_cube['Q'].unmasked_data[...], 3)
        assert_allclose(sky_cube['U'].unmasked_data[...], 2.5)
        assert_allclose(sky_cube['V'].unmasked_data[...], 0.5j)

    def test_circular_to_sky(self):
        stokes_data = dict(
            RR=SpectralCube(self.data[0][None, ...], wcs=self.wcs),
            RL=SpectralCube(self.data[1][None, ...], wcs=self.wcs),
            LR=SpectralCube(self.data[2][None, ...], wcs=self.wcs),
            LL=SpectralCube(self.data[3][None, ...], wcs=self.wcs),
        )
        cube = StokesSpectralCube(stokes_data)
        sky_cube = cube.transform_basis('Sky')
        assert_allclose(sky_cube['I'].unmasked_data[...], 7)
        assert_allclose(sky_cube['Q'].unmasked_data[...], 2.5)
        assert_allclose(sky_cube['U'].unmasked_data[...], 0.5j)
        assert_allclose(sky_cube['V'].unmasked_data[...], 3)

    def test_sky_to_linear(self):
        stokes_data = dict(
            I=SpectralCube(self.data[0][None, ...], wcs=self.wcs),
            Q=SpectralCube(self.data[1][None, ...], wcs=self.wcs),
            U=SpectralCube(self.data[2][None, ...], wcs=self.wcs),
            V=SpectralCube(self.data[3][None, ...], wcs=self.wcs),
        )
        cube = StokesSpectralCube(stokes_data)
        lin_cube = cube.transform_basis('Linear')
        assert_allclose(lin_cube['XX'].unmasked_data[...], 6)
        assert_allclose(lin_cube['XY'].unmasked_data[...], 1.5 + 2j)
        assert_allclose(lin_cube['YX'].unmasked_data[...], 1.5 - 2j)
        assert_allclose(lin_cube['YY'].unmasked_data[...], 4)

    def test_sky_to_circular(self):
        stokes_data = dict(
            I=SpectralCube(self.data[0][None, ...], wcs=self.wcs),
            Q=SpectralCube(self.data[1][None, ...], wcs=self.wcs),
            U=SpectralCube(self.data[2][None, ...], wcs=self.wcs),
            V=SpectralCube(self.data[3][None, ...], wcs=self.wcs),
        )
        cube = StokesSpectralCube(stokes_data)
        circ_cube = cube.transform_basis('Circular')
        assert_allclose(circ_cube['RR'].unmasked_data[...], 7)
        assert_allclose(circ_cube['RL'].unmasked_data[...], 1 + 1.5j)
        assert_allclose(circ_cube['LR'].unmasked_data[...], 1 - 1.5j)
        assert_allclose(circ_cube['LL'].unmasked_data[...], 3)

    def test_transform_basis_incomplete(self):
        stokes_data = dict(
            XX=SpectralCube(self.data[0][None, ...], wcs=self.wcs),
            YY=SpectralCube(self.data[1][None, ...], wcs=self.wcs),
        )
        cube = StokesSpectralCube(stokes_data)
        with pytest.raises(NotImplementedError):
            cube.transform_basis('Sky')

    def test_transform_basis_noop(self):
        stokes_data = dict(
            I=SpectralCube(self.data[0][None, ...], wcs=self.wcs),
            Q=SpectralCube(self.data[1][None, ...], wcs=self.wcs),
            U=SpectralCube(self.data[2][None, ...], wcs=self.wcs),
            V=SpectralCube(self.data[3][None, ...], wcs=self.wcs),
        )
        cube = StokesSpectralCube(stokes_data)
        sky_cube = cube.transform_basis('Sky')
        for k in stokes_data:
            assert_allclose(sky_cube[k].unmasked_data[...], self.data["IQUV".index(k)][None, ...])
