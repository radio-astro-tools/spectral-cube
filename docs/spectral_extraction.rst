Spectral Extraction
===================

A commonly required operation is extracting a spectrum from a part of a cube.

The simplest way to get a spectrum from the cube is simply to slice it along
a single pixel::

    >>> spectrum = cube[:, 50, 60]  # doctest: +SKIP

Slicing along the first dimension will create a
`~spectral_cube.lower_dimensional_structures.OneDSpectrum` object, which has a few
useful capabilities.

Aperture Extraction
-------------------

Going one level further, you can extract a spectrum from an aperture.
We'll start with the simplest variant: a square aperture.  The
cube can be sliced in pixel coordinates to produce a sub-cube
which we then average spatially to get the spectrum::

    >>> subcube = cube[:, 50:53, 60:63]  # doctest: +SKIP
    >>> spectrum = subcube.mean(axis=(1,2))  # doctest: +SKIP

The spectrum can be obtained using any mathematical operation, such as the
``max`` or ``std``, e.g., if you wanted to obtain the noise spectrum.

Slightly more sophisticated aperture extraction
-----------------------------------------------

To get the flux in a circular aperture, you need to mask the data.  In this
example, we don't use any external libraries, but show how to create a circular
mask from scratch and apply it to the data.::

    >>> yy, xx = np.indices([5,5], dtype='float')  # doctest: +SKIP
    >>> radius = ((yy-2)**2 + (xx-2)**2)**0.5  # doctest: +SKIP
    >>> mask = radius <= 2  # doctest: +SKIP
    >>> subcube = cube[:, 50:55, 60:65]  # doctest: +SKIP
    >>> maskedsubcube = subcube.with_mask(mask)  # doctest: +SKIP
    >>> spectrum = maskedsubcube.mean(axis=(1,2))  # doctest: +SKIP

Aperture and spectral extraction using regions
----------------------------------------------

Spectral-cube supports ds9 and crtf regions, so you can use them to create a
mask.  The ds9/crtf region support relies on `regions
<https://astropy-regions.readthedocs.io/en/latest/>`_, which supports most
shapes in ds9 and crtf, so you are not limited to circular apertures.

In this example, we'll extract a subcube from ds9 region string using
:meth:`~spectral_cube.BaseSpectralCube.subcube_from_ds9region`::

    >>> ds9_str = 'fk5; circle(19:23:43.907, +14:30:34.66, 3")'  # doctest: +SKIP
    >>> subcube = cube.subcube_from_ds9region(ds9_str)  # doctest: +SKIP
    >>> spectrum = subcube.mean(axis=(1, 2))  # doctest: +SKIP

Similarly, one can extract a subcube from a crtf region string using
:meth:`~spectral_cube.BaseSpectralCube.subcube_from_crtfregion`::

    >>> crtf_str = 'circle[[19:23:43.907, +14:30:34.66], 3"], coord=fk5, range=[150km/s, 300km/s]'  # doctest: +SKIP
    >>> subcube = cube.subcube_from_crtfregion(crtf_str)  # doctest: +SKIP
    >>> spectrum = subcube.mean(axis=(1, 2))  # doctest: +SKIP

You can also use a _list_ of `~regions.Region` objects to extract a subcube using
:meth:`~spectral_cube.BaseSpectralCube.subcube_from_regions`::

    >>> import regions # doctest: +SKIP
    >>> regpix = regions.RectanglePixelRegion(regions.PixCoord(0.5, 1), width=4, height=2)  # doctest: +SKIP
    >>> subcube = cube.subcube_from_regions([regpix])  # doctest: +SKIP
    >>> spectrum = subcube.mean(axis=(1, 2))  # doctest: +SKIP

To learn more, go to :ref:`reg`.
