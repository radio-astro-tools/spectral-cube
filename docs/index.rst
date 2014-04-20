Spectral Cube documentation
===========================

The ``spectral_cube`` package provides an easy way to read, manipulate,
analyze, and write data cubes with two positional dimensions and one
spectral dimension, optionally with Stokes parameters. The spectral cube
reader is designed to be robust to the wide range of conventions of axis
order, spatial projections, and spectral units that exist, and provides a
uniform interface to all spectral cubes. The package is designed to work with
large files, including files larger than the available memory on a computer.

Using ``spectral_cube``
-----------------------

The package centers around the
:class:`~spectral_cube.SpectralCube` class. In the following
sections, we look at how to read data into this class, manipulate spectral
cubes, extract moment maps or subsets of spectral cubes, and write spectral
cubes to files.

.. toctree::
   :maxdepth: 1

   creating_reading.rst
   accessing.rst
   subsets.rst
   moments.rst
   masking.rst
   stokes.rst
   big_data.rst

.. automodapi:: spectral_cube
   :no-inheritance-diagram:
