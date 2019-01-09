Metadata and Headers
====================

The metadata of both :class:`~spectral_cube.SpectralCube` s and
:class:`~spectral_cube.lower_dimensional_structures.LowerDimensionalObject` s is
stored in their ``.meta`` attribute, which is a dictionary of metadata.

When writing these objects to file, or exporting them as FITS :class:`HDUs
<astropy.io.fits.PrimaryHDU>`, the metadata will be written to the FITS header.
If the metadata matches the FITS standard, it will just be directly written,
with the dictionary keys replaced with upper-case versions.  If the keys are
longer than 8 characters, a FITS ``COMMENT`` entry will be entered with the
data formatted as ``{key}={value}``.

The world coordinate system (WCS) metadata will be handled automatically, as
will the beam parameter metadata.  The automation implies that WCS keywords
and beam keywords cannot be manipulated directly by changing the ``meta``
dictionary; they must be manipulated through other means (e.g.,
:doc:`manipulating`).


