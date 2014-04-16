#!/usr/bin/env python

import sys
if 'build_sphinx' in sys.argv or 'develop' in sys.argv:
    from setuptools import setup, Command
else:
    from distutils.core import setup, Command

with open('README.md') as file:
    long_description = file.read()

with open('CHANGES') as file:
    long_description += file.read()

# no versions yet from agpy import __version__ as version

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)

execfile('spectral_cube/version.py')

setup(name='spectral_cube',
      version=__version__,
      description='Spectral Cube object',
      long_description=long_description,
      author='Chris Beaumont, Adam Leroy, Adam Ginsburg, Erik Rosolowsky, and Tom Robitaille',
      author_email='adam.g.ginsburg@gmail.com',
      url='https://github.com/radio_tools/spectral_cube',
      packages=['spectral_cube','spectral_cube.io','spectral_cube.tests'], 
      cmdclass = {'test': PyTest},
     )
