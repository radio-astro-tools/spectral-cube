Installing ``spectral-cube``
============================

Requirements
------------

This package has the following dependencies:

* `Python <http://www.python.org>`_ Python 3.x
* `Numpy <http://www.numpy.org>`_ 1.8 or later
* `Astropy <http://www.astropy.org>`__ 4.0 or later
* `radio_beam <https://github.com/radio-astro-tools/radio_beam>`_, used when
  reading in spectral cubes that use the BMAJ/BMIN convention for specifying the beam size.
* `Bottleneck <http://berkeleyanalytics.com/bottleneck/>`_, optional (speeds
  up median and percentile operations on cubes with missing data)
* `Regions <https://astropy-regions.readthedocs.io/en/latest>`_ >=0.3dev, optional
  (Serialises/Deserialises DS9/CRTF region files and handles them. Used when
  extracting a subcube from region)
* `dask <https://dask.org/>`_, used for the :class:`~spectral_cube.DaskSpectralCube` class
* `zarr <https://zarr.readthedocs.io/en/stable/>`_ and `fsspec <https://pypi.org/project/fsspec/>`_,
  used for storing computations to disk when using the dask-enabled classes.
* `six <http://pypi.python.org/pypi/six/>`_
* `casa-formats-io <https://pypi.org/project/casa-formats-io>`_

Installation
------------

To install the latest stable release, you can type::

    pip install spectral-cube

or you can download the latest tar file from
`PyPI <https://pypi.python.org/pypi/spectral-cube>`_ and install it using::

    python setup.py install

If you are using python2.7 (e.g., if you are using CASA version 5 or earlier),
the latest spectral-cube version that is compatible is v0.4.4.

Developer version
-----------------

If you want to install the latest developer version of the spectral cube code, you
can do so from the git repository::

    git clone https://github.com/radio-astro-tools/spectral-cube.git
    cd spectral-cube
    python setup.py install

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/3/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install git+https://github.com/radio-astro-tools/spectral-cube.git

Installing into CASA
--------------------
Installing packages in CASA is fairly straightforward.  The process is described `here <http://docs.astropy.org/en/stable/install.html#installing-astropy-into-casa>`_.  In short, you can do the following:

First, we need to make sure `pip <https://pypi.python.org/pypi/pip>`__ is
installed. Start up CASA as normal, and type::

    CASA <1>: from setuptools.command import easy_install

    CASA <2>: easy_install.main(['--user', 'pip'])

Now, quit CASA and re-open it, then type the following to install ``spectral-cube``::

    CASA <1>: import subprocess, sys

    CASA <2>: subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'spectral-cube'])


For CASA versions 5 and earlier, you need to install a specific version of spectral-cube because more recent
versions of spectral-cube require python3.::

    CASA <1>: import subprocess, sys

    CASA <2>: subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'spectral-cube==v0.4.4'])
