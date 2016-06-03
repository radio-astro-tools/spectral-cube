#!/usr/bin/env python
import pip
import sys
pip.main(['install', 'numpy', '--user'])
pip.main(['install', 'astropy', '--user'])

print("astropy installed")

sys.exit(0)
