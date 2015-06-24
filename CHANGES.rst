0.2.3 (unreleased)
------------------

 - Add experimental line-finding tool using astroquery.splatalogue
   (https://github.com/radio-astro-tools/spectral-cube/pull/210)

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
