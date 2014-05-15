Spectral Cube documentation
===========================

The ``spectral_cube`` package provides an easy way to read, manipulate,
analyze, and write data cubes with two positional dimensions and one
spectral dimension, optionally with Stokes parameters.

``spectral_cube`` aims to be a versatile data container for building
custom analysis routines. It provides the following main features:

- A uniform interface to spectral cubes, robust to the
  wide range of conventions of axis order, spatial projections,
  and spectral units that exist in the wild.
- Easy extraction of cube sub-regions using physical coordinates.
- Ability to easily create, combine, and apply masks to datasets.
- Basic summary statistic methods like moments and array aggregates.
- Designed to work with datasets too large to load into memory.

Quick start
-----------

Here's a simple script demonstrating ``spectral_cube``::

    >>> import astropy.units as u
    >>> from spectral_cube import SpectralCube
    >>> cube = SpectralCube.read('data.fits')
    >>> print cube
    SpectralCube with shape=(563, 640, 640) and unit=K:
    n_x: 640  type_x: RA---SIN  unit_x: deg
    n_y: 640  type_y: DEC--SIN  unit_y: deg
    n_s: 563  type_s: FREQ      unit_s: Hz

    # extract the subcube between 98 and 100 GHz
    >>> slab = cube.spectral_slab(98 * u.GHz, 100 * u.GHz)

    # Ignore elements fainter than 1K
    >>> masked_slab = slab.with_mask(slab > 1)

    # Compute the first moment and write to file
    >>> m1 = masked_slab.moment(order=1)
    >>> m1.write('moment_1.fits')

Using ``spectral_cube``
-----------------------

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
   masking.rst
   accessing.rst
   subsets.rst
   moments.rst

Advanced
^^^^^^^^

.. toctree::
   :maxdepth: 1

   .. stokes.rst
   yt_example.rst
   big_data.rst
   api.rst
