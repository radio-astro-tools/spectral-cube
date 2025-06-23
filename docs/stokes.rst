:orphan:

StokesSpectralCube: Handling Stokes Components
==============================================

The `StokesSpectralCube` class in `spectral_cube` provides a convenient and robust way to handle spectral cubes with multiple Stokes components (e.g., I, Q, U, V, XX, YY, RR, LL, etc.).

Features
--------
- **Multiple Stokes Axes:** Store and access cubes for each Stokes component in a unified object.
- **Astropy Integration:** Uses `astropy.coordinates.StokesCoord` for validation and mapping of Stokes axes, including support for custom symbols.
- **Basis Transformations:** Convert between sky (I, Q, U, V), linear feed (XX, XY, YX, YY), and circular feed (RR, RL, LR, LL) representations using `transform_basis`.
- **Masking:** Supports global and per-component masks, with robust logic for combining masks.
- **Tested:** Comprehensive test suite covers all major features and edge cases.

Basic Usage
-----------

.. code-block:: python

    from spectral_cube import SpectralCube, StokesSpectralCube
    from astropy.wcs import WCS
    import numpy as np

    # Example WCS and data
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'FREQ']
    data = np.random.randn(4, 5, 5)  # 4 Stokes components

    # Create SpectralCube objects for each Stokes axis
    stokes_data = {
        'I': SpectralCube(data[0][None, ...], wcs=wcs),
        'Q': SpectralCube(data[1][None, ...], wcs=wcs),
        'U': SpectralCube(data[2][None, ...], wcs=wcs),
        'V': SpectralCube(data[3][None, ...], wcs=wcs),
    }
    cube = StokesSpectralCube(stokes_data)

    # Access components
    I_cube = cube['I']
    Q_cube = cube.Q  # attribute access also works

    # Transform to linear feed basis
    lin_cube = cube.transform_basis('Linear')
    XX_cube = lin_cube['XX']

    # Transform to circular feed basis
    circ_cube = cube.transform_basis('Circular')
    RR_cube = circ_cube['RR']

    # Combine with a mask
    from spectral_cube.masks import BooleanArrayMask
    mask = BooleanArrayMask(np.random.rand(1, 5, 5) > 0.5, wcs=wcs)
    masked_cube = cube.with_mask(mask)

Supported Stokes Axes
---------------------

The following Stokes axes are supported (including custom symbols):

- I, Q, U, V (sky)
- XX, XY, YX, YY (linear feed)
- RR, RL, LR, LL (circular feed)
- RX, RY, LX, LY, XR, XL, YR, YL, PP, PQ, QP, QQ (custom axes): values = -9 to -20
- RCircular, LCircular, Linear, Ptotal, Plinear, PFtotal, PFlinear, Pangle (custom axes): values = -21 to -28
  (See the code for the full list and mapping.)

Basis Transformations
---------------------

Use `transform_basis` to convert between supported bases:

.. code-block:: python

    # Linear to sky
    sky_cube = lin_cube.transform_basis('Sky')
    # Circular to sky
    sky_cube = circ_cube.transform_basis('Sky')
    # Sky to linear
    lin_cube = cube.transform_basis('Linear')
    # Sky to circular
    circ_cube = cube.transform_basis('Circular')

Limitations
-----------
- All cubes must have the same shape and WCS.
- Transformations require all four components for the basis.
- Partial or mixed bases are not yet supported.

See Also
--------
- :class:`spectral_cube.SpectralCube`
- :class:`astropy.coordinates.StokesCoord`

.. note::
   This class is an experimental implementation and is tested. Please report any issues or feature requests on the project issue tracker.