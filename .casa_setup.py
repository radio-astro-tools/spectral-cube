#!/usr/bin/env python
from setuptools.command import easy_install
easy_install.main(['--user', 'pip'])

import pip
pip.main(['install', 'astropy', '--user'])

import astropy
