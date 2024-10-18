import warnings
import inspect

from functools import wraps

from astropy.utils.exceptions import AstropyUserWarning

bigdataurl = "https://spectral-cube.readthedocs.io/en/latest/big_data.html"

from tqdm.auto import tqdm

def ProgressBar(niter, **kwargs):
    return tqdm(total=niter, **kwargs)


def cached(func):
    """
    Decorator to cache function calls
    """

    @wraps(func)
    def wrapper(self, *args):
        # The cache lives in the instance so that it gets garbage collected
        if (func, args) not in self._cache:
            self._cache[(func, args)] = func(self, *args)
        return self._cache[(func, args)]

    wrapper.wrapped_function = func

    return wrapper

def warn_slow(function):

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        # if the function accepts a 'how', the 'cube' approach requires the whole cube in memory
        argspec = inspect.getfullargspec(function)
        accepts_how_keyword = 'how' in argspec.args or argspec.varkw == 'how'

        warn_how = accepts_how_keyword and ((kwargs.get('how') == 'cube') or 'how' not in kwargs)

        # This restores showing the "loading the entire cube into memory" warning for
        # _apply_everywhere and _cube_on_cube_operation
        if function.__name__ in ['_apply_everywhere', '_cube_on_cube_operation']:
            warn_how = True

        if self._is_huge and not self.allow_huge_operations:
            warn_message = ("This function ({0}) requires loading the entire "
                            "cube into memory, and the cube is large ({1} "
                            "pixels), so by default we disable this operation. "
                            "To enable the operation, set "
                            "`cube.allow_huge_operations=True` and try again.  ").format(str(function), self.size)

            if warn_how:
                warn_message += ("Alternatively, you may want to consider using an "
                                 "approach that does not load the whole cube into "
                                 "memory by specifying how='slice' or how='ray'.  ")

            warn_message += ("See {bigdataurl} for details.".format(bigdataurl=bigdataurl))

            raise ValueError(warn_message)
        elif warn_how and not self._is_huge:
            # TODO: add check for whether cube has been loaded into memory
            warnings.warn("This function ({0}) requires loading the entire cube into "
                          "memory and may therefore be slow.".format(str(function)),
                          PossiblySlowWarning
                         )
        return function(self, *args, **kwargs)
    return wrapper

class SpectralCubeWarning(AstropyUserWarning):
    pass

class UnsupportedIterationStrategyWarning(SpectralCubeWarning):
    pass

class VarianceWarning(SpectralCubeWarning):
    pass

class SliceWarning(SpectralCubeWarning):
    pass

class BeamAverageWarning(SpectralCubeWarning):
    pass

class BeamWarning(SpectralCubeWarning):
    pass

class WCSCelestialError(Exception):
    pass

class WCSMismatchWarning(SpectralCubeWarning):
    pass

class NotImplementedWarning(SpectralCubeWarning):
    pass

class StokesWarning(SpectralCubeWarning):
    pass

class ExperimentalImplementationWarning(SpectralCubeWarning):
    pass

class PossiblySlowWarning(SpectralCubeWarning):
    pass

class SmoothingWarning(SpectralCubeWarning):
    pass

class NonFiniteBeamsWarning(SpectralCubeWarning):
    pass

class WCSWarning(SpectralCubeWarning):
    pass

class FITSWarning(SpectralCubeWarning):
    pass

class BadVelocitiesWarning(SpectralCubeWarning):
    pass

class FITSReadError(Exception):
    pass

class NoBeamError(Exception):
    pass

class BeamUnitsError(Exception):
    pass
