#!/usr/bin/env python
import sys
from setuptools.command import easy_install

#easy_install.main(['--user', 'hashlib-20081119.zip'])
easy_install.main(['--user', 'urllib3-1.21.tar.gz'])
easy_install.main(['--user', 'pip-9.0.1.tar.gz'])
# fails because CASA wasn't compiled against good versions of hash/sha lib
#easy_install.main(['--user', 'pip'])

print("pip installed.")

sys.exit(0)
