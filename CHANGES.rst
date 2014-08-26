0.2 (unreleased)
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
- Added `copy` and `deepcopy` methods for masks (https://github.com/radio-astro-tools/spectral-cube/pull/127)

0.1 (2014-06-01)
----------------

- Initial Release.
