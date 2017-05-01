#!/usr/bin/env python
import astropy
import sys

from astropy.io import fits
import numpy as np

print("astropy imported successfully.  "
      "Version: {0}, path: {1}".format(astropy.__version__, astropy.__path__))
print("numpy imported successfully.  "
      "Version: {0}, path: {1}".format(numpy.__version__, numpy.__path__))

sys.exit(0)
