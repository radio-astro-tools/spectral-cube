#!/usr/bin/env python
import pip
import sys
#pip.main(['install', '--user', 'numpy-1.12.1.zip'])
pip.main(['install', '--user', 'astropy-1.3.2.tar.gz'])
#pip.main(['install', 'numpy', '--user'])
#pip.main(['install', 'astropy', '--user'])

print("astropy installed")

sys.exit(0)
