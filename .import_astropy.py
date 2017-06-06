#!/usr/bin/env python
import sys
import os

paths = os.getenv('PYPATH').split(":")

for pp in paths:
    sys.path.insert(0, pp)

import astropy

from astropy.io import fits
import numpy as np

print('sys.path: {0}'.format(sys.path))
print("astropy imported successfully.  "
      "Version: {0}, path: {1}".format(astropy.__version__, astropy.__path__))
print("numpy imported successfully.  "
      "Version: {0}, path: {1}".format(np.__version__, np.__path__))
print("Numpy has nanmedian: {0}".format(hasattr(np, 'nanmedian')))

sys.exit(0)
