Quick Looks
===========

Once you've loaded a cube, you inevitably will want to look at it in various
ways.  Slices in any direction have `quicklook` methods:

    >>> cube[50,:,:].quicklook() # show an image
    >>> cube[:, 50, 50].quicklook() # plot a spectrum

The same can be done with moments:

    >>> cube.moment0(axis=0).quicklook()

PVSlicer
--------
The `pvextractor <http://pvextractor.readthedocs.org/en/latest/>`_ package
comes with a GUI that has a simple matplotlib image viewer.  To activate it
for your cube:

    >>> cube.to_pvextractor()
