Spectral Cube documentation
===========================

The spectral-cube package provides an easy way to read, manipulate, analyze,
and write data cubes with two positional dimensions and one spectral dimension,
optionally with Stokes parameters. It provides the following main features:

- A uniform interface to spectral cubes, robust to the
  wide range of conventions of axis order, spatial projections,
  and spectral units that exist in the wild.
- Easy extraction of cube sub-regions using physical coordinates.
- Ability to easily create, combine, and apply masks to datasets.
- Basic summary statistic methods like moments and array aggregates.
- Designed to work with datasets too large to load into memory.

Quick start
-----------

Here's a simple script demonstrating the spectral-cube package::

    >>> import astropy.units as u
    >>> from spectral_cube import SpectralCube
    >>> cube = SpectralCube.read('adv.fits')  # doctest: +SKIP
    >>> print cube  # doctest: +SKIP
    SpectralCube with shape=(4, 3, 2) and unit=K:
    n_x:      2  type_x: RA---SIN  unit_x: deg    range:    24.062698 deg:   24.063349 deg
    n_y:      3  type_y: DEC--SIN  unit_y: deg    range:    29.934094 deg:   29.935209 deg
    n_s:      4  type_s: VOPT      unit_s: m / s  range:  -321214.699 m / s: -317350.054 m / s

    # extract the subcube between 98 and 100 GHz
    >>> slab = cube.spectral_slab(98 * u.GHz, 100 * u.GHz)  # doctest: +SKIP

    # Ignore elements fainter than 1K
    >>> masked_slab = slab.with_mask(slab > 1)  # doctest: +SKIP

    # Compute the first moment and write to file
    >>> m1 = masked_slab.moment(order=1)  # doctest: +SKIP
    >>> m1.write('moment_1.fits')  # doctest: +SKIP

Using spectral-cube
-------------------

The package centers around the
:class:`~spectral_cube.SpectralCube` class. In the following
sections, we look at how to read data into this class, manipulate spectral
cubes, extract moment maps or subsets of spectral cubes, and write spectral
cubes to files.

Getting started
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   installing.rst
   creating_reading.rst
   accessing.rst
   masking.rst
   arithmetic.rst
   manipulating.rst
   smoothing.rst
   writing.rst
   moments.rst
   errors.rst
   quick_looks.rst
   beam_handling.rst
   spectral_extraction.rst

Advanced
^^^^^^^^

.. toctree::
   :maxdepth: 1

   yt_example.rst
   big_data.rst
   api.rst
