#!/usr/bin/env python

import os
import sys

import numpy

from setuptools import setup
from setuptools.extension import Extension

TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:

    tox -e test

If you don't already have tox installed, you can install it with:

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .
    pytest

For more information, see:

  http://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if 'test' in sys.argv:
    print(TEST_HELP)
    sys.exit(1)

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:

    tox -e build_docs

If you don't already have tox installed, you can install it with:

    pip install tox

For more information, see:

  http://docs.astropy.org/en/latest/install.html#builddocs
"""

if 'build_docs' in sys.argv or 'build_sphinx' in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)

setup(use_scm_version={'write_to': os.path.join('spectral_cube', 'version.py')},
      ext_modules=[Extension("spectral_cube.io._casa_chunking",
                             [os.path.join('spectral_cube', 'io', '_casa_chunking.c')],
                             include_dirs=[numpy.get_include()])])
