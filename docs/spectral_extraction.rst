Spectral Extraction
===================

A commonly required operation is extracting a spectrum from a part of a cube.

The simplest way to get a spectrum from the cube is simply to slice it along
a single pixel:

    >>> spectrum = cube[:, 50, 60]

Slicing along the first dimension will create a
`~spectral_cube.lower_dimensional_structures.OneDSpectrum` object, which has a few
useful capabilities.

Aperture Extraction
-------------------

Going one level further, you can extract a spectrum from an aperture.
We'll start with the simplest variant: a square aperture.  The
cube can be sliced in pixel coordinates to produce a sub-cube
which we then average spatially to get the spectrum:

    >>> subcube = cube[:, 50:53, 60:63]
    >>> spectrum = subcube.mean(axis=(1,2))

The spectrum can be obtained using any mathematical operation, such as the
``max`` or ``std``, e.g., if you wanted to obtain the noise spectrum.

Slightly more sophisticated aperture extraction
-----------------------------------------------

To get the flux in a circular aperture, you need to mask the data.  In this
example, we don't use any external libraries, but show how to create a circular
mask from scratch and apply it to the data.

    >>> yy, xx = np.indices([5,5], dtype='float')
    >>> radius = ((yy-2)**2 + (xx-2)**2)**0.5
    >>> mask = radius <= 2
    >>> subcube = cube[:, 50:55, 60:65]
    >>> maskedsubcube = subcube.with_mask(mask)
    >>> spectrum = maskedsubcube.mean(axis=(1,2))

Aperture extraction using regions
---------------------------------

Spectral-cube supports ds9 regions, so you can use the ds9 region to create a
mask.  The ds9 region support relies on `pyregion
<https://pyregion.readthedocs.io/en/latest/>`_, which supports most shapes in
ds9, so you are not limited to circular apertures.

In this example, we'll create a region "from scratch", but you can also use a
predefined region file using `pyregion.open
<http://pyregion.readthedocs.io/en/latest/api/pyregion.open.html>`_.

    >>> shapelist = pyregion.parse("fk5; circle(19:23:43.907,+14:30:34.66, 3\")")
    >>> subcube = cube.subcube_from_ds9region(shapelist)
    >>> spectrum = subcube.mean(axis=(1,2))

Eventually, we hope to change the region support from pyregion to `astropy
regions <http://astropy-regions.readthedocs.io/en/latest/>`_, so the
above example may become obsolete.
