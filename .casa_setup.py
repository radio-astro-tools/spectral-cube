import sys
import os

paths = os.getenv('PYPATH').split(":")

for pp in paths:
    sys.path.insert(0, pp)

exec(open("./setup.py").read())
