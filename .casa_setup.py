import sys
import os

if os.getenv('PYPATH'):
    paths = os.getenv('PYPATH').split(":")
else:
    print("PYPATH not specified.")
    sys.exit(1)

for pp in paths:
    sys.path.insert(0, pp)


import numpy as np
print("numpy imported successfully.  "
      "Version: {0}, path: {1}".format(np.__version__, np.__path__))
print("Numpy has nanmedian: {0}".format(hasattr(np, 'nanmedian')))

print(sys.argv)
# pop the 'extra' arguments
print(sys.argv.pop(0)) # casapy
print(sys.argv.pop(0)) # nogui
print(sys.argv.pop(0)) # nologger
print(sys.argv.pop(0)) # -c

try:
    from taskinit import ia
    print("Successfully recognized CASA environment")
except ImportError:
    print("Failed to recognize CASA environment")
    sys.exit(1)

exec(open("./setup.py").read())
