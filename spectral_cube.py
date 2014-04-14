import numpy as np
import wcs_manipulation

class SpectralCubeMask(object):
    
    def __init__(self, wcs, mask, include=True):
        self._wcs = wcs
        self._includemask = mask if include else np.logical_not(mask)
        
    @property
    def include(self):
        return self._includemask

    @property
    def exclude(self):
        return np.logical_not(self._includemask)

class SpectralCube(object):
    
    def __init__(self, data, wcs, mask, metadata={}):
        self._wcs = wcs
        self._data = data
        self._spectral_axis = None
        self._mask = mask  # specifies which elements to Nan/blank/ignore -> SpectralCubeMask
                           # object or array-like object, given that WCS needs to be consistent with data?
        assert mask._wcs == self._wcs
        self.metadata = metadata

    def _transpose_axes(self):
        # PLACEHOLDER:
        # we likely will not want to run this but use the logic without any array manips
        axtypes = self._wcs.get_axis_types()
        for ii,ax in enumerate(axtypes):
            if ax['coordinate_type'] == 'spectral':
                specaxisnumber = ii

        if specaxisnumber != 0:
            self._data = self._data.swapaxes(specaxisnumber, 0)
            self._wcs = wcs_manipulation.wcs_swapaxes(self._wcs, specaxisnumber, 0)

    @property
    def shape(self):
        return self._data.shape

    def read(self, filename, format=None):
        pass
        
    def write(self, filename, format=None):
        pass
    
    def sum(self, axis=None):
        pass
    
    def max(self, axis=None):
        pass
    
    def min(self, axis=None):
        pass
    
    def argmax(self, axis=None):
        pass
    
    def argmin(self, axis=None):
        pass
    
    def interpolate_slice(self):
        # Find a slice at an exact spectral value?
        pass
    
    @property
    def data_valid(self):
        """ Flat array of unmasked data values """
        return self._data[self.mask.include]
    
    def chunked(self, chunksize=1000):
        """
        Iterate over chunks of valid data
        """
        yield blah
        
    @property
    def data(self):
        """
        Return the underlying data as a numpy array.
        Always returns the spectral axis as the 0th axis
        
        Sets masked values to NaN or 0 (whatever is more useful)
        """

    @property
    def data_unmasked(self):
        """
        Like data, but don't apply the mask
        """
        
    def data_filled(replacement):
        """Behaves like .data but replaces masked values with `replacement`
        """
        
    @property
    def spectral_axis(self):
        """
        A `~astropy.units.Quantity` array containing the central values of each channel
        """
        # TODO: use world[...] once implemented
        iz, iy, ix = np.broadcast_arrays(np.arange(self.shape[0]), 0., 0.)
        return self._wcs.all_pix2world(ix, iy, iz, 0)[2]

    def apply_mask(self, mask, inherit=True):
        """
        Return a new Cube object, that applies the input mask to the underlying data.
        
        Handles any necessary logic to resample the mask
        
        If inherit=True, the new Cube uses the union of the original and new mask
        
        What type of object is `mask`? SpectralMask?
        Not sure -- I guess it needs to have WCS. Conceptually maps onto a boolean ndarray
            CubeMask? -> maybe SpectralCubeMask to be consistent with SpectralCube
            
        """
        
    def moment(self, order, wcs=False):
        """
        Determine the n'th moment along the spectral axis
        
        If *wcs = True*, return the WCS describing the moment
        """
    
    def spectral_slab(self, low, high, restfreq=None):
        """
        Need better name - extract a new cube between two spectral values
    
        lo, hi can be quantitites, to "do the right thing" with regards
        to velocity/wavelength/frequency. restfreq might be needed for this
        """
        spectral = self.spectral_axis
        #if low.
    
    def world_spines(self):
        """
        Returns a dict of 1D arrays, for the world coordinates
        along each pixel axis. 
        
        Raises error if this operation is ill-posed (e.g. rotated world coordinates,
        strong distortions)
        """
        
    @property
    def world(self):
        """
        Access the world coordinates for the cube, as if it was a Numpy array, so:
        
        >>> cube.world[0,:,:]
        
        returns a dictionary of 2-d arrays giving the coordinates in the first spectral slice.
        
        This can be made to only compute the required values rather than compute everything then slice.
        """

    

# demo code
def test():
    x = SpectralCube()
    x2 = SpectralCube()
    x.sum(axis='spectral')
    # optionally:
    x.sum(axis=0) # where 0 is defined to be spectral
    x.moment(3) # kurtosis?  moment assumes spectral
    (x*x2).sum(axis='spatial1')
