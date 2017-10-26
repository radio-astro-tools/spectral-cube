0.4.2 (unreleased)
------------------
 - Bugfix and enhancement: handle multiple beams using radio_beam's
   multiple-beams feature.  This allows `convolve_to` to work when some beams
   are masked out.
   (https://github.com/radio-astro-tools/spectral-cube/pull/437)

0.4.1 (2017-10-17)
------------------
 - Add SpectralCube.with_beam and Projection.with_beam for attaching
   beam objects. Raise error for position-spectral slices of VRSCs
   (https://github.com/radio-astro-tools/spectral-cube/pull/433)
 - Raise a nicer error if no data is present in the default or
   selected HDU
   (https://github.com/radio-astro-tools/spectral-cube/pull/424)
 - Check mask inputs to OneDSpectrum and add mask handling for
   OneDSpectrum.spectral_interpolate
   (https://github.com/radio-astro-tools/spectral-cube/pull/400)
 - Improve exception if cube does not have two celestial and one
   spectral dimesnion
   (https://github.com/radio-astro-tools/spectral-cube/pull/425)
 - Add creating a Projection from a FITS HDU
   (https://github.com/radio-astro-tools/spectral-cube/pull/376)
 - Deprecate numpy <=1.8 because nanmedian is needed
   (https://github.com/radio-astro-tools/spectral-cube/pull/373)
 - Add tools for masking bad beams in VaryingResolutionSpectralCubes
   (https://github.com/radio-astro-tools/spectral-cube/pull/373)
 - Don't warn if no beam was found in a cube
   (https://github.com/radio-astro-tools/spectral-cube/pull/422)

0.4.0 (2016-09-06)
------------------
 - Handle equal beams when convolving cubes spatially.
   (https://github.com/radio-astro-tools/spectral-cube/pull/356)
 - Whole cube convolution & reprojection has been added, including tools to
   smooth spectrally and spatially to force two cubes onto an identical grid.
   (https://github.com/radio-astro-tools/spectral-cube/pull/313)
 - Bugfix: files larger than the available memory are now readable again
   because ``spectral-cube`` does not encourage you to modify cubes inplace
   (https://github.com/radio-astro-tools/spectral-cube/pull/299)
 - Cube planes with bad beams will be masked out
   (https://github.com/radio-astro-tools/spectral-cube/pull/298)
 - Added a new cube type, VaryingResolutionSpectralCube, meant to handle
   CASA-produced cubes that have different beams in each channel
   (https://github.com/radio-astro-tools/spectral-cube/pull/292)
 - Added tests for new functionality in OneDSpectrum
   (https://github.com/radio-astro-tools/spectral-cube/pull/277)
 - Split out common functionality between SpectralCube and LowerDimensionalObject
   into BaseNDClass and SpectralAxisMixinClass
   (https://github.com/radio-astro-tools/spectral-cube/pull/274)
 - added new linewidth_sigma and linewidth_fwhm methods to SpectralCube for
   computing linewidth maps, and make sure the documentation is clear that
   moment(order=2) is a variance map.
   (https://github.com/radio-astro-tools/spectral-cube/pull/275)
 - fixed significant error when the cube WCS includes a cd matrix.  This
   error resulted in incorrect spectral coordinate conversions
   (https://github.com/radio-astro-tools/spectral-cube/pull/276)

0.3.2 (2016-07-11)
------------------

 - Bugfix in configuration

0.3.1 (2016-02-04)
------------------

 - Preserve metadata when making projections
   (https://github.com/radio-astro-tools/spectral-cube/pull/250)
 - bugfix: cube._data cannot be a quantity
   (https://github.com/radio-astro-tools/spectral-cube/pull/251)
 - partial fix for ds9 import bug
   (https://github.com/radio-astro-tools/spectral-cube/pull/253)
 - preserve WCS information in projections
   (https://github.com/radio-astro-tools/spectral-cube/pull/256)
 - whitespace stripped from BUNIT
   (https://github.com/radio-astro-tools/spectral-cube/pull/257)
 - bugfix: sometimes cube would be read into memory when it should not be
   (https://github.com/radio-astro-tools/spectral-cube/pull/259)
 - more projection preservation fixes
   (https://github.com/radio-astro-tools/spectral-cube/pull/265)
 - correct jy/beam capitalization
   (https://github.com/radio-astro-tools/spectral-cube/pull/267)
 - convenience attribute for beam access
   (https://github.com/radio-astro-tools/spectral-cube/pull/268)
 - fix beam reading, which would claim failure even during success
   (https://github.com/radio-astro-tools/spectral-cube/pull/271)

0.3.0 (2015-08-16)
------------------

 - Add experimental line-finding tool using astroquery.splatalogue
   (https://github.com/radio-astro-tools/spectral-cube/pull/210)
 - Bugfixes (211,212,217)
 - Add arithmetic operations (add, subtract, divide, multiply, power)
   (https://github.com/radio-astro-tools/spectral-cube/pull/220).
   These operations will not be permitted on large cubes by default, but will
   require the user to specify that they are allowed using the attribute
   ``allow_huge_operations``
 - Implemented slicewise stddev and mean
   (https://github.com/radio-astro-tools/spectral-cube/pull/225)
 - Bugfix: prevent a memory leak when creating a large number of Cubes
   (https://github.com/radio-astro-tools/spectral-cube/pull/233)
 - Provide a ``base`` attribute so that tools like joblib can operate on
   ``SpectralCube`` s as memory maps
   (https://github.com/radio-astro-tools/spectral-cube/pull/230)
 - Masks have a quicklook method
   (https://github.com/radio-astro-tools/spectral-cube/pull/228)
 - Memory mapping can be disabled
   (https://github.com/radio-astro-tools/spectral-cube/pull/226)
 - Add xor operations for Masks
   (https://github.com/radio-astro-tools/spectral-cube/pull/241)
 - Added a new StokesSpectralCube class to deal with 4-d cubes
   (https://github.com/radio-astro-tools/spectral-cube/pull/249)

0.2.2 (2015-03-12)
------------------

- Output mask as a CASA image https://github.com/radio-astro-tools/spectral-cube/pull/171
- ytcube exports to .obj and .ply too
  https://github.com/radio-astro-tools/spectral-cube/pull/173
- Fix air wavelengths, which were mistreated
  (https://github.com/radio-astro-tools/spectral-cube/pull/186)
- Add support for sum/mean/std over both spatial axes to return a
  OneDSpectrum object.  This PR also removes numpy 1.5-1.7 tests, since
  many `spectral_cube` functions are not compatible with these versions
  of numpy (https://github.com/radio-astro-tools/spectral-cube/pull/188)

0.2.1 (2014-12-03)
------------------

- CASA cube readers now compatible with ALMA .image files (tested on Cycle 2
  data) https://github.com/radio-astro-tools/spectral-cube/pull/165
- Spectral quicklooks available
  https://github.com/radio-astro-tools/spectral-cube/pull/164 now that 1D
  slices are possible
  https://github.com/radio-astro-tools/spectral-cube/pull/157
- `to_pvextractor` tool allows easy export to `pvextractor
  <pvextractor.readthedocs.org>`_
  https://github.com/radio-astro-tools/spectral-cube/pull/160
- `to_glue` sends the cube to `glue <www.glueviz.org/en/latest/>`_
  https://github.com/radio-astro-tools/spectral-cube/pull/153


0.2 (2014-09-11)
----------------

- `moments` preserve spectral units now https://github.com/radio-astro-tools/spectral-cube/pull/118
- Initial support added for Air Wavelength.  This is only 1-way support,
  round-tripping (vacuum->air) is not supported yet.
  https://github.com/radio-astro-tools/spectral-cube/pull/117
- Integer slices (single frames) are supported
  https://github.com/radio-astro-tools/spectral-cube/pull/113
- Bugfix: BUNIT capitalized https://github.com/radio-astro-tools/spectral-cube/pull/112
- Masks can be any array that is broadcastable to the cube shape
  https://github.com/radio-astro-tools/spectral-cube/pull/115
- Added `.header` and `.hdu` convenience methods https://github.com/radio-astro-tools/spectral-cube/pull/120
- Added public functions `apply_function` and `apply_numpy_function` that allow
  functions to be run on cubes while preserving important metadata (e.g., WCS)
- Added a quicklook tool using aplpy to view slices (https://github.com/radio-astro-tools/spectral-cube/pull/131)
- Added subcube and ds9 region extraction tools (https://github.com/radio-astro-tools/spectral-cube/pull/128)
- Added a `to_yt` function for easily converting between SpectralCube and yt
  datacube/dataset objects
  (https://github.com/radio-astro-tools/spectral-cube/pull/90,
  https://github.com/radio-astro-tools/spectral-cube/pull/129)
- Masks' `.include()` method works without ``data`` arguments.
  (https://github.com/radio-astro-tools/spectral-cube/pull/147)
- Allow movie name to be specified in yt movie creation
  (https://github.com/radio-astro-tools/spectral-cube/pull/145)
- add `flattened_world` method to get the world coordinates corresponding to
  each pixel in the flattened array
  (https://github.com/radio-astro-tools/spectral-cube/pull/146)

0.1 (2014-06-01)
----------------

- Initial Release.
