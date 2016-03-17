Manipulating cubes
==================

Modifying the spectral axis
---------------------------

As mentioned in :doc:`accessing`, it is straightforward to find the
coordinates along the spectral axis using the
:attr:`~spectral_cube.SpectralCube.spectral_axis` attribute::

   >>> cube.spectral_axis  # doctest: +SKIP
   [ -2.97198762e+03  -2.63992044e+03  -2.30785327e+03  -1.97578610e+03
     -1.64371893e+03  -1.31165176e+03  -9.79584583e+02  -6.47517411e+02
     ...
      3.15629983e+04   3.18950655e+04   3.22271326e+04   3.25591998e+04
      3.28912670e+04   3.32233342e+04] m / s

The default units of a spectral axis are determined from the FITS header or
WCS object used to initialize the cube, but it is also possible to change the
spectral axis unit using :meth:`~spectral_cube.SpectralCube.with_spectral_unit`::

    >>> from astropy import units as u
    >>> cube2 = cube.with_spectral_unit(u.km / u.s)  # doctest: +SKIP
    >>> cube2.spectral_axis  # doctest: +SKIP
    [ -2.97198762e+00  -2.63992044e+00  -2.30785327e+00  -1.97578610e+00
      -1.64371893e+00  -1.31165176e+00  -9.79584583e-01  -6.47517411e-01
      ...
       3.02347296e+01   3.05667968e+01   3.08988639e+01   3.12309311e+01
       3.15629983e+01   3.18950655e+01   3.22271326e+01   3.25591998e+01
       3.28912670e+01   3.32233342e+01] km / s

It is also possible to change from velocity to frequency for example, but
this requires specifying the rest frequency or wavelength as well as a
convention for the doppler shift calculation::

    >>> cube3 = cube.with_spectral_unit(u.GHz, velocity_convention='radio',
    ...                                 rest_value=200 * u.GHz)  # doctest: +SKIP
    [ 220.40086492  220.40062079  220.40037667  220.40013254  220.39988841
      220.39964429  220.39940016  220.39915604  220.39891191  220.39866778
      ...
      220.37645231  220.37620818  220.37596406  220.37571993  220.3754758
      220.37523168  220.37498755  220.37474342  220.3744993   220.37425517] GHz

The new cubes will then preserve the new spectral units when computing
moments for example (see :doc:`moments`).

Extracting a spectral slab
--------------------------

Given a spectral cube, it is easy to extract a sub-cube covering only a subset
of the original range in the spectral axis. To do this, you can use the
:meth:`~spectral_cube.SpectralCube.spectral_slab` method. This
method takes lower and upper bounds for the spectral axis, as well as an
optional rest frequency, and returns a new
:class:`~spectral_cube.SpectralCube` instance. The bounds can
be specified as a frequency, wavelength, or a velocity but the units have to
match the type of the spectral units in the cube (if they do not match, first
use :meth:`~spectral_cube.SpectralCube.with_spectral_unit` to ensure that they
are in the same units). The bounds should be given as Astropy
:class:`Quantities <astropy.units.Quantity>` as follows::

    >>> from astropy import units as u
    >>> subcube = cube.spectral_slab(-50 * u.km / u.s, +50 * u.km / u.s)  # doctest: +SKIP

The resulting cube ``subcube`` (which is also a
:class:`~spectral_cube.SpectralCube` instance) then contains all channels
that overlap with the range -50 to 50 km/s relative to the rest frequency
assumed by the world coordinates, or the rest frequency specified by a prior
call to :meth:`~spectral_cube.SpectralCube.with_spectral_unit`.

Extracting a sub-cube by indexing
---------------------------------

It is also easy to extract a sub-cube from pixel coordinates using standard
Numpy slicing notation::

    >>> sub_cube = cube[:100, 10:50, 10:50]  # doctest: +SKIP

This returns a new :class:`~spectral_cube.SpectralCube` object
with updated WCS information.

Extracting a subcube from a ds9 region
--------------------------------------

Starting with spectral_cube v0.2, you can use ds9 regions to extract subcubes.
The minimal enclosing subcube will be extracted with a two-dimensional mask
corresponding to the ds9 region.  `pyregion
<http://leejjoon.github.io/pyregion/>`_ is required for region parsing::

    >>> region_list = pyregion.open('file.reg')  # doctest: +SKIP
    >>> sub_cube = cube.subcube_from_ds9region(region_list)  # doctest: +SKIP

If you want to loop over individual regions with a single region file, you need to convert the individual
region to a shape list due to limitations in pyregion::

  >>> region_list = pyregion.open('file.reg')  #doctest: +SKIP
  >>> for region in region_list: #doctest: +SKIP
  >>>     sub_cube = cube.subcube_from_ds9region(pyregion.ShapeList([region])) #doctest: +SKIP
    
You can also create a region on the fly using ds9 region syntax.  This extracts
a 0.1 degree circle around the Galactic Center::

    >>> region_list = pyregion.parse("galactic; circle(0,0,0.1)")  # doctest: +SKIP
    >>> sub_cube = cube.subcube_from_ds9region(region_list)  # doctest: +SKIP

    
Extract the minimal valid subcube
---------------------------------

If you have a mask that masks out some of the cube edges, such that the
resulting sub-cube might be smaller in memory, it can be useful to extract the
minimal enclosing sub-cube::

    >>> sub_cube = cube.minimal_subcube()  # doctest: +SKIP

You can also shrink any cube by this mechanism::

    >>> sub_cube = cube.with_mask(smaller_region).minimal_subcube()  # doctest: +SKIP
