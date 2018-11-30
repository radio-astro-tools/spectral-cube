Visualization
=============

Spectral-cube is not primarily a visualization package, but it has several
tools for visualizing subsets of the data.

All lower-dimensional subsets,
`~spectral_cube.lower_dimensional_structures.OneDSpectrum`, and
`~spectral_cube.lower_dimensional_structures.Projection`, have their own
``quicklook`` methods
(`~spectral_cube.lower_dimensional_structures.OneDSpectrum.quicklook` and
`~spectral_cube.lower_dimensional_structures.Projection.quicklook`,
respectively).  These methods will plot the data with somewhat properly labeled
axes.

The two-dimensional viewers default to using `aplpy <http://aplpy.github.io/>`_.
Because of quirks of how aplpy sets up its plotting window, these methods will
create their own figures.  If ``use_aplpy`` is set to ``False``, and similarly
if you use the ``OneDSpectrum`` quicklook, the data will be overplotted in the
latest used plot window.


In principle, one can also simply plot the data.  For example, if you have a cube,
you could do::

    >>> plt.plot(cube[:,0,0]) # doctest: +SKIP

to plot a spectrum sliced out of the cube or::

    >>> plt.imshow(cube[0,:,:]) # doctest: +SKIP

to plot an image slice. 

.. warning:: 

   There are known incompatibilities with the above plotting approach:
   matplotlib versions ``<2.1`` will crash, and you will have to clear the plot
   window to reset it.


Other Visualization Tools
=========================
To visualize the cubes directly, you can use some of the other tools we provide
for pushing cube data into external viewers.

See :doc:`yt_example` for using yt as a visualization tool.


The `spectral_cube.SpectralCube.to_glue` and
`spectral_cube.SpectralCube.to_ds9` methods will send the whole cube to glue
and ds9.  This approach generally requires loading the whole cube into memory.
