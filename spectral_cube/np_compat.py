import numpy as np
from distutils.version import StrictVersion

def allbadtonan(function):
    """
    Wrapper of numpy's nansum etc.: for <=1.8, just return the function's
    results.  For >=1.9, any axes with all-nan values will have all-nan outputs
    in the collapsed version
    """
    def f(data, axis=-1):
        result = function(data, axis=axis)
        if StrictVersion(np.__version__) >= StrictVersion('1.9.0'):
            if axis is None:
                if np.all(np.isnan(data)):
                    return np.nan
                else:
                    return result
            nans = np.all(np.isnan(data), axis=axis)
            result[nans] = np.nan
        return result

    return f
