#!/usr/bin/env python
import sys
from setuptools.command import easy_install

easy_install.main(['--user', 'pip-9.0.1.tar.gz'])
# fails because CASA wasn't compiled against good versions of hash/sha lib
#easy_install.main(['--user', 'pip'])

print("pip installed.")

sys.exit(0)
