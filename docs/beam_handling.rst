Handling Beams
==============

If you are using radio data, your cubes should have some sort of beam
information included.  ``spectral-cube`` handles beams using the `radio_beam
<https://github.com/radio-astro-tools/radio_beam>`_
package.

There are two ways beams can be stored in FITS files: as FITS header
keywords (``BMAJ``, ``BMIN``, and ``BPA``) or as a ``BinTableHDU``
extension.  If the latter is present, ``spectral-cube`` will return
a `~spectral_cube.spectral_cube.VaryingResolutionSpectralCube` object.

For the simpler case of a single beam across all channels, the presence
of the beam allows for direct conversion of a cube with Jy/beam units
to surface brightness (K) units.  Note, however, that this requires
loading the entire cube into memory!::

   >>> cube.unit  # doctest: +SKIP
   Unit("Jy / beam")
   >>> kcube = cube.to(u.K)
   >>> kcube.unit
   Unit("K")

Multi-beam cubes
----------------

Varying resolution (multi-beam) cubes are somewhat trickier to work with in
general, though unit conversion is easy.  You can perform the same sort of unit
conversion with `~spectral_cube.spectral_cube.VaryingResolutionSpectralCube`s
as with regular `~spectral_cube.spectral_cube.SpectralCube`s; ``spectral-cube``
will use a different beam and frequency for each plane.

For other sorts of operations, discussion of how to deal with these cubes via
smoothing to a common resolution is in the `smoothing` document.
