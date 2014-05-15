Accessing and modifying the spectral axis
=========================================

Given a cube object, it is straightforward to find the coordinates along the spectral axis::

   >>> cube.spectral_axis
   [ -2.97198762e+03  -2.63992044e+03  -2.30785327e+03  -1.97578610e+03
     -1.64371893e+03  -1.31165176e+03  -9.79584583e+02  -6.47517411e+02
     ...
      3.15629983e+04   3.18950655e+04   3.22271326e+04   3.25591998e+04
      3.28912670e+04   3.32233342e+04] m / s

The default units of a spectral axis are determined from the FITS header or
WCS object used to initialize the cube, but it is also possible to change the
spectral axis unit using :meth:`~spectral_cube.SpectralCube.with_spectral_unit`::

    >>> from astropy import units as u
    >>> cube2 = cube.with_spectral_unit(u.km / u.s)
    >>> cube2.spectral_axis
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
                                        rest_value=200 * u.GHz)
    [ 220.40086492  220.40062079  220.40037667  220.40013254  220.39988841
      220.39964429  220.39940016  220.39915604  220.39891191  220.39866778
      ...
      220.37645231  220.37620818  220.37596406  220.37571993  220.3754758
      220.37523168  220.37498755  220.37474342  220.3744993   220.37425517] GHz

The new cubes will then preserve the new spectral units when computing
moments for example.
