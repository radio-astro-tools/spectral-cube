import sys
import os

paths = os.getenv('PYPATH').split(":")

for pp in paths:
    sys.path.insert(0, pp)

import numpy as np
print("numpy imported successfully.  "
      "Version: {0}, path: {1}".format(np.__version__, np.__path__))
print("Numpy has nanmedian: {0}".format(hasattr(np, 'nanmedian')))

exec(open("./setup.py").read())
