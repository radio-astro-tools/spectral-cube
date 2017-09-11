Continuum Subtraction
=====================

A common task with data cubes is continuum identification and subtraction.  For
line-rich cubes where the continuum is difficult to identify, you should use
`statcont <https://github.com/radio-astro-tools/statcont>`_.
For single-line cubes, the process is much easier.

First, the simplest case is when you have a single line that makes up
a small fraction of the total observed band, e.g., a narrow line.
In this case, you can use a simple median approximation for the continuum.::

    >>> med = cube.median(axis=0)  # doctest: +SKIP
    >>> med_sub_cube = cube - med  # doctest: +SKIP

The second part of this task may complain that the cube is too big.  If it
does, you can still do the above operation by first setting
``cube.allow_huge_operations=True``, but be warned that this can be expensive.

For a more complicated case, you may want to mask out the line-containing
channels.  This can be done using a spectral boolean mask.::

    >>> from astropy import units as u  # doctest: +SKIP
    >>> import numpy as np  # doctest: +SKIP
    >>> spectral_axis = cube.with_spectral_unit(u.km/u.s).spectral_axis  # doctest: +SKIP
    >>> good_channels = (spectral_axis < 25*u.km/u.s) | (spectral_axis > 45*u.km/u.s)  # doctest: +SKIP
    >>> masked_cube = cube.with_mask(good_channels[:, np.newaxis, np.newaxis])  # doctest: +SKIP
    >>> med = masked_cube.median(axis=0)  # doctest: +SKIP
    >>> med_sub_cube = cube - med  # doctest: +SKIP

The array ``good_channels`` is a simple 1D numpy boolean array that is ``True``
for all channels below 25 km/s and above 45 km/s, and is ``False`` for all
channels in the range 25-45 km/s.  The indexing trick ``good_channels[:,
np.newaxis, np.newaxis]`` (or equivalently, ``good_channels[:, None, None]``)
is just a way to tell the cube which axes to project along.  In more
recent versions of ``spectral-cube``, the indexing trick is not necessary.
The median in this case is computed only over the specified line-free channels.

Any operation can be used to compute the continuum, such as the ``mean`` or
some ``percentile``, but for most use cases, the ``median`` is fine.
