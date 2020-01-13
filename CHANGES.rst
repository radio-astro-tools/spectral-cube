0.5 (unreleased)
----------------

- Refactor tests to use fixtures for accessing data instead of needing to
  run a script to generate test files. #598

0.4.5 (unreleased)
------------------
 - Added support for casatools-based io in #541 and beam reading from CASA
   images in #543
 - Add support for ``update_function`` in the joblib-based job distributor
   in #534
 - Add tests for WCS equivalence in reprojected images in #589
 - Improve error messages when CASA files are read incorrectly in #584
 - fix a small bug in matplotlib figure saving in #583
 - Allow for reading of beamless cubes in #582
 - Add support for 2d world functions in #575 and extrema in #552
 - Handle kernels defined as quantities in smoothing in #578
 - Fix bug with NPOL header keyword in #576
 - Convolution will be skipped if beans are equal-sized in #573
 - Fix one-D sliceing with no beam in #568
 - Paralellization documentation improvement in #557
 - Astropy-helpers updated to 2.0.10 in #553
 - Fixed some future warnings in #565
 - Added a new documentation example in #548
 - Added channel map making capability in #551
 - Fix warnings when beam is not defined in #561
 - Improvment to joblib parallelization in #564
 - Add ppbeam attribute to lower-dimensional objects #549
 - Handle CASA file beams in #543 and #545
 - Add support for CASA reading using casatools (casa6) in #541
 - Bugfix for slicing of different shapes in #532
 - Fixes to yt integratino in #531
 - Add `unmasked_beams` attribute and change many beams behaviors in #502
 - Bugfix for downsampled WCS corners in #525
 - Performance enhancement to world extrema in #524
 - Simplify conversion from CASA coordsys to FITS-WCS #593
 - Add chunked file reading for CASA .image opening #592
 - Dropped python 3.5 testing in #592

0.4.4 (2019-02-20)
------------------
 - Refactor all beam parameters into mix-in classes; added BaseOneDSpectrum
   for common functionality between OneDSpectrum and VaryingResolutionOneDSpectrum.
   Retain beam objects when doing arithmetic with LDOs/
   (https://github.com/radio-astro-tools/spectral-cube/pull/521)
 - Refactor OneDSpectrum objects to include a single beam if they
   were produced from a cube with a single beam to enable K<->Jy
   conversions
   (https://github.com/radio-astro-tools/spectral-cube/pull/510)
 - Bugfix: fix compatibility of to_glue with latest versions of glue.
   (https://github.com/radio-astro-tools/spectral-cube/pull/491)
 - Refactor to use regions instead of pyregion.  Adds CRTF support
   (https://github.com/radio-astro-tools/spectral-cube/pull/488)
 - Direct downsampling tools added, both in-memory and memmap
   (https://github.com/radio-astro-tools/spectral-cube/pull/486)

0.4.3 (2018-04-05)
------------------
 - Refactor spectral smoothing tools to allow parallelized application *and*
   memory mapped output (to avoid loading cube into memory).  Created
   ``apply_function_parallel_spectral`` to make this general.  Added
   ``joblib`` as a dependency.
   (https://github.com/radio-astro-tools/spectral-cube/pull/474)
 - Bugfix: Reversing a cube's spectral axis should now do something reasonable
   instead of unreasonable
   (https://github.com/radio-astro-tools/spectral-cube/pull/478)

0.4.2 (2018-02-21)
------------------
 - Bugfix and enhancement: handle multiple beams using radio_beam's
   multiple-beams feature.  This allows `convolve_to` to work when some beams
   are masked out.  Also removes ``cube_utils.average_beams``, which is now
   implemented directly in radio_beam
   (https://github.com/radio-astro-tools/spectral-cube/pull/437)
 - Added a variety of stacking tools, both for stacking full velocity
   cubes of different lines and for stacking full spectra based on
   a velocity field (https://github.com/radio-astro-tools/spectral-cube/pull/446,
   https://github.com/radio-astro-tools/spectral-cube/pull/453,
   https://github.com/radio-astro-tools/spectral-cube/pull/457,
   https://github.com/radio-astro-tools/spectral-cube/pull/465)

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
