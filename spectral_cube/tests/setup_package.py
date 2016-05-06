from __future__ import print_function, absolute_import, division

def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['coveragerc', 'data/*.fits', 'data/*.hdr', 'data/*.lmv', 'data/*reg']
    }
