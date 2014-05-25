Installing ``spectral_cube``
============================

Requirements
------------

This package has the following dependencies:

* `Python <http://www.python.org>`_ 2.6 or later (Python 3.x is supported)
* `Numpy <http://www.numpy.org>`_ 1.5.1 or later
* `Astropy <http://www.astropy.org>`_ 0.3.0 or later
* `Bottleneck <http://berkeleyanalytics.com/bottleneck/>`_, optional (speeds
  up median and percentile operations on cubes with missing data)

Installation
------------

To install the latest stable release, you can type::

    pip install spectral_cube

or you can download the latest tar file from
`PyPI <https://pypi.python.org/pypi/spectral_cube>`_ and install it using::

    python setup.py install

Developer version
-----------------

If you want to install the latest developer version of the spectral cube code, you
can do so from the git repository::

    git clone https://github.com/radio-astro-tools/spectral-cube.git
    cd spectral-cube
    python setup.py install

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install https://github.com/radio-astro-tools/spectral-cube/archive/master.zip
