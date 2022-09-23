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

You can find the latest version and the issue tracker on `github
<https://github.com/radio-astro-tools/spectral-cube>`_.

Quick start
-----------

Here's a simple script demonstrating the spectral-cube package::

    >>> import astropy.units as u
    >>> from astropy.utils import data
    >>> from spectral_cube import SpectralCube
    >>> fn = data.get_pkg_data_filename('tests/data/example_cube.fits', 'spectral_cube')
    >>> cube = SpectralCube.read(fn)
    >>> print(cube)
    SpectralCube with shape=(7, 4, 3) and unit=Jy / beam:
     n_x:      3  type_x: RA---ARC  unit_x: deg    range:    52.231466 deg:   52.231544 deg
     n_y:      4  type_y: DEC--ARC  unit_y: deg    range:    31.243639 deg:   31.243739 deg
     n_s:      7  type_s: VRAD      unit_s: m / s  range:    14322.821 m / s:   14944.909 m / s

    # extract the subcube between 98 and 100 GHz
    >>> slab = cube.spectral_slab(98 * u.GHz, 100 * u.GHz)  # doctest: +SKIP

    # Ignore elements fainter than 1 Jy/beam
    >>> masked_slab = slab.with_mask(slab > 1 Jy/beam)  # doctest: +SKIP

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
   :maxdepth: 2

   installing.rst
   creating_reading.rst
   accessing.rst

Cube Analysis
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   moments.rst
   errors.rst
   writing.rst
   beam_handling.rst
   masking.rst
   arithmetic.rst
   metadata.rst
   smoothing.rst
   reprojection.rst

Subsets
^^^^^^^

.. toctree::
   :maxdepth: 2

   manipulating.rst
   spectral_extraction.rst
   continuum_subtraction.rst

Visualization
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   quick_looks.rst
   visualization.rst


Other Examples
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   examples.rst

There is also an `astropy tutorial
<http://learn.astropy.org/rst-tutorials/FITS-cubes.html>`__ on accessing and
manipulating FITS cubes with spectral-cube.

Advanced
^^^^^^^^

.. toctree::
   :maxdepth: 1

   dask.rst
   yt_example.rst
   big_data.rst
   developing_with_spectralcube.rst
   api.rst
