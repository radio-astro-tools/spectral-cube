from __future__ import print_function, absolute_import, division

import numpy as np
from distutils.version import LooseVersion

def allbadtonan(function):
    """
    Wrapper of numpy's nansum etc.: for <=1.8, just return the function's
    results.  For >=1.9, any axes with all-nan values will have all-nan outputs
    in the collapsed version
    """
    def f(data, axis=None, keepdims=None):
        if keepdims is None:
            result = function(data, axis=axis)
        else:
            result = function(data, axis=axis, keepdims=keepdims)
        if LooseVersion(np.__version__) >= LooseVersion('1.9.0') and hasattr(result, '__len__'):
            if axis is None:
                if np.all(np.isnan(data)):
                    return np.nan
                else:
                    return result
            if keepdims is None:
                nans = np.all(np.isnan(data), axis=axis)
            else:
                nans = np.all(np.isnan(data), axis=axis, keepdims=keepdims)
            result[nans] = np.nan
        return result

    return f
