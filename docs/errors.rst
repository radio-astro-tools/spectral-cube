.. doctest-skip-all


Explanations of commonly-encountered error messages
===================================================

Beam parameters differ
----------------------

If you are using spectral cubes produced by CASA's tclean, it may have a
different *beam size* for each channel.  In this case, it will be loaded as a
`~spectral_cube.VaryingResolutionSpectralCube` object.  If you perform any
operations spanning the spectral axis, for example ``cube.moment0(axis=0)`` or
``cube.max(axis=0)``, you may encounter errors like this one:

.. code::

    Beam srs differ by up to 1.0x, which is greater than the threshold 0.01.

This occurs if the beam sizes are different by more than the specified
threshold factor.  A default threshold of 1% is set because for most
interferometric images, beam differences on this scale are negligible (they
correspond to flux measurement errors of 10^-4).  

To inspect the beam properties, look at the ``beams`` attribute, for example:

.. code::

   >>> cube.beams
   [Beam: BMAJ=1.1972888708114624 arcsec BMIN=1.0741511583328247 arcsec BPA=72.71219635009766 deg,
    Beam: BMAJ=1.1972869634628296 arcsec BMIN=1.0741279125213623 arcsec BPA=72.71561431884766 deg,
    Beam: BMAJ=1.1972919702529907 arcsec BMIN=1.0741302967071533 arcsec BPA=72.71575164794922 deg,
    ...
    Beam: BMAJ=1.1978825330734253 arcsec BMIN=1.0744788646697998 arcsec BPA=72.73623657226562 deg,
    Beam: BMAJ=1.1978733539581299 arcsec BMIN=1.0744799375534058 arcsec BPA=72.73489379882812 deg,
    Beam: BMAJ=1.197875738143921 arcsec BMIN=1.0744699239730835 arcsec BPA=72.73745727539062 deg]

In this example, the beams differ by a tiny amount that is below the threshold.
However, sometimes you will encounter cubes with dramatically different beam
sizes, and spectral-cube will prevent you from performing operations along the
spectral axis with these beams because such operations are poorly defined.

There are several options to manage this problem:

  1. Increase the threshold.  This is best done if the beams still differ by a
     small amount, but larger than 1%.  To do this, set ``cube.beam_threshold =
     [new value]``.  This is the `"tape over the check engine light"
     <https://www.youtube.com/watch?v=ddPQAJSm2cQ>`_ approach; use with caution.
  2. Convolve the cube to a common resolution using
     `~spectral_cube.SpectralCube.convolve_to`.  This is again best if the largest
     beam is only slightly larger than the smallest.
  3. Mask out the bad channels.  For example:

.. code::

   good_beams = cube.identify_bad_beams(threshold=0.1)
   mcube = cube.mask_out_bad_beams(threshold=0.1)




Moment-2 or FWHM calculations give unexpected NaNs
--------------------------------------------------

It is fairly common to have moment 2 calculations return NaN values along
pixels where real values are expected, e.g., along pixels where both moment0
and moment1 return real values.

Most commonly, this is caused by "bad baselines", specifically, by large sections
of the spectrum being slightly negative at large distances from the centroid position
(the moment 1 position).  Because moment 2 weights pixels at larger distances more
highly (as the square of the distance), slight negative values at large distances
can result in negative values entering the square root when computing the line width
or the FWHM.

The solution is either to make a tighter mask, excluding the pixels far from
the centroid position, or to ensure that the baseline does not have any
negative systematic offset.


Looking at images with matplotlib
---------------------------------
Matplotlib accesses a lot of hidden properties of arrays when plotting.  If you
try to show a slice with ``imshow``, you may encounter the WCS-related error::

    NotImplementedError: Reversing an axis is not implemented.

If you see this error, the only solution at present is to specify
``origin='lower'``, which is the standard for images anyway.  For example::

    import pylab as pl
    pl.imshow(cube[5,:,:], origin='lower')

should work, where ``origin='upper'`` will not.  This is due to a limitation in
``astropy.wcs`` slicing.

An alternative option, if it is absolutely necessary to use ``origin='upper'``
or if you encounter other matplotlib-related issues, is to use the ``.value``
attribute of the slice to get a bald numpy array to plot::

    import pylab as pl
    pl.imshow(cube[5,:,:].value)
