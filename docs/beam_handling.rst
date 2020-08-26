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
general, though some operations are similar to a single resolution cube.
Unit conversion is one case where the operation is called the same for
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube` s
as with regular `~spectral_cube.spectral_cube.SpectralCube` s. For example, to
convert from Jy / beam to K::

    >>> vrsc_cube_K = vrsc_cube.to(u.K)  # doctest: +SKIP

``spectral-cube`` will use a different beam and frequency for each plane.

Handling variations in the beams is often a source of difficulty. Some spectral
operations (e.g., moment maps) require a common resolution. However, a few channels
may have large discprepancies in the beam compared to most of the others (e.g., if
more data in a particular channel was bad and had to be flagged).
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube` deals with these problems in two ways.

Finding and convolving to a common resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When applying spectral operations on a
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube`, it is often
necessary to first convolve the data to have a common beam. To retain the
maximum amount of spatial information, the data can be convolved to the
smallest common beam that all beams in
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.beam` can be
deconvolved from.

Finding the smallest common beam and convolution operations are described
in detail in :doc:`smoothing`.


Small beam variations: Do I need to convolve first?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Formally, operations on multiple spectral channels require the data to have a common
resolution. In practice, data cubes with a high-spectral resolution will have the beam
vary by a very small factor between adjacent channels. It is then a safe approximation
to ignore small changes in the beam to avoid having to convolve the entire data cube to
a common beam (which might be a slow and expensive operation for large cubes).

`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube` allows for small variations
in the beam by setting
:meth:`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.beam_threshold`. The
`beam_threshold` is the largest allowed fractional change relative to the smallest common beam that will be allowed in spectral operations. For example, the default 0.01 allows up
to a 1% variation from the smallest common beam. If the beams vary more than this, a
`ValueError` will be raised prior to the spectral operation. Otherwise, variations between
the beams will be ignored, a warning will be printed describing this approximation, and
the spectral operation will be computed.

The beam threshold can be changed from the default 0.01 with::

    >>> vrsc_cube.beam_threshold = 0.02   # doctest: +SKIP

.. note::

    Setting `vrsc_cube.beam_threshold = 1.0` will allow all beam variations without raising an error. Large values for the beam threshold should not be used for scientific results.

For most spectral-line data cubes covering a single spectral line, the fine resolution
and small frequency range will often allow the above approximation to work.
However, if you are working with wideband data, this approximation will not hold and you
will need to convolve to a common beam, or extract spectral slabs from the data cube.

Strict mode
***********

To disable allowing small variations in the beam,
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube` has a strict checking mode
than can be enabled with::

    >>> vrsc_cube.strict_beam_mode = True  # doctest: +SKIP

When strict mode is enabled, all beam must be identical for spectral operations to work.
Note that this is equivalent to setting the beam threshold to 0.


Identifying channels with bad beams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some cubes may have channels with beams that vary drastically over small ranges overlap
channels. This is often the case where a range of channels has poor data or is affected
by radio frequency interference, leading to most of the data in that channel being flagged.
If these channels are kept, the smallest common beam (see :doc:`smoothing`) may be much
larger due to these channels.

You can identify and mask channels with bad beams using
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.identify_bad_beams`::

    >>> goodbeams = vrsc_cube.identify_bad_beams()  # doctest: +SKIP

This will return a 1D boolean mask where `True` means the channel beam is good.
By default,
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.identify_bad_beams`
will use
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.beam_threshold` (default
of 0.01; see above).
However, the comparison here is to the **median** major and minor axis rather than
the smallest common beam used above.
This is because bad beams are identified as outliers in the set of beams.

To mask the channels with bad beams, use
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.mask_out_bad_beams`.


    >>> masked_vrsc_cube = vrsc_cube.mask_out_bad_beams()  # doctest: +SKIP

The masked cube without the bad beams will now exlude channels with bad beams and
can be used, for example, to convolve to a better representative common beam
resolution (see above).

We also note that, in general, you can mask out individual channels using
`~spectral_cube.spectral_cube.VaryingResolutionSpectralCube.mask_channels`.

.. note::

    The common beam is not used to find discrepant and bad beams since they are
    identified as outliers from the set. We note that this is an approximate method of
    finding channels with outlier beams and may fail in some cases. Please
    `raise an issue <https://github.com/radio-astro-tools/spectral-cube/issues>`_ if
    this method does not work well for your data cube.
