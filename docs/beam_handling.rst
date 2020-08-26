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
   >>> kcube = cube.to(u.K)  # doctest: +SKIP
   >>> kcube.unit  # doctest: +SKIP
   Unit("K")


Adding a Beam
-------------

If your cube does not have a beam, a custom beam can be attached given::

    >>> new_beam = Beam(1. * u.deg)  # doctest: +SKIP
    >>> new_cube = cube.with_beam(new_beam)  # doctest: +SKIP
    >>> new_cube.beam  # doctest: +SKIP
    Beam: BMAJ=3600.0 arcsec BMIN=3600.0 arcsec BPA=0.0 deg

This is handy for synthetic observations, which initially have a point-like beam::

    >>> point_beam = Beam(0 * u.deg)  # doctest: +SKIP
    >>> new_cube = synth_cube.with_beam(point_beam)  # doctest: +SKIP
    Beam: BMAJ=0.0 arcsec BMIN=0.0 arcsec BPA=0.0 deg

The cube can then be convolved to a new resolution::

    >>> new_beam = Beam(60 * u.arcsec)  # doctest: +SKIP
    >>> conv_synth_cube = synth_cube.convolve_to(new_beam)  # doctest: +SKIP
    >>> conv_synth_cube.beam  # doctest: +SKIP
    Beam: BMAJ=60.0 arcsec BMIN=60.0 arcsec BPA=0.0 deg

Beam can also be attached in the same way for `~spectral_cube.Projection` and
`~spectral_cube.Slice` objects.

Multi-beam cubes
----------------

Varying resolution (multi-beam) cubes are somewhat trickier to work with in
general, though unit conversion is easy.  You can perform the same sort of unit
conversion with `~spectral_cube.spectral_cube.VaryingResolutionSpectralCube` s
as with regular `~spectral_cube.spectral_cube.SpectralCube` s. For example, to
convert from Jy / beam to K::

    >>> vrsc_cube_K = vrsc_cube.to(u.K)  # doctest: +SKIP

``spectral-cube`` will use a different beam and frequency for each plane.

You can identify channels with bad beams (i.e., beams that differ from a reference beam,
which by default is the median beam) using
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.identify_bad_beams`
(the returned value is a mask array where ``True`` means the channel is good),
mask channels with undesirable beams using
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.mask_out_bad_beams`,
and in general mask out individual channels using
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.mask_channels`.

For other sorts of operations, discussion of how to deal with these cubes via
smoothing to a common resolution is in the :doc:`smoothing` document.
