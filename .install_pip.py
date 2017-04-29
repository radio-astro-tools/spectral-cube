#!/usr/bin/env python
import sys
from setuptools.command import easy_install
easy_install.main(['--user', 'hashlib'])
easy_install.main(['--user', 'pip'])

print("pip installed.")

sys.exit(0)
