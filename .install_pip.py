#!/usr/bin/env python
import sys
from setuptools.command import easy_install
easy_install.main(['--user', 'pip'])
sys.exit(0)
