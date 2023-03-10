from astropy import units as u
from astropy import log
import numpy as np
import warnings
import abc

import astropy
from astropy.io.fits import Card
from radio_beam import Beam, Beams
import dask.array as da

from . import wcs_utils
from . import cube_utils
from .utils import BeamWarning, cached, WCSCelestialError, BeamAverageWarning, NoBeamError, BeamUnitsError
from .masks import BooleanArrayMask


__doctest_skip__ = ['SpatialCoordMixinClass.world']
__all__ = ['BaseNDClass', 'BeamMixinClass',
           'HeaderMixinClass', 'MaskableArrayMixinClass',
           'MultiBeamMixinClass', 'SpatialCoordMixinClass',
           'SpectralAxisMixinClass',
          ]

DOPPLER_CONVENTIONS = {}
DOPPLER_CONVENTIONS['radio'] = u.doppler_radio
DOPPLER_CONVENTIONS['optical'] = u.doppler_optical
DOPPLER_CONVENTIONS['relativistic'] = u.doppler_relativistic


class BaseNDClass(object):

    _cache = {}

    @property
    def _nowcs_header(self):
        """
        Return a copy of the header with no WCS information attached
        """
        log.debug("Stripping WCS from header")
        return wcs_utils.strip_wcs_from_header(self._header)

    @property
    def wcs(self):
        return self._wcs

    @property
    def meta(self):
        return self._meta

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value


class HeaderMixinClass(object):
    """
    A mixin class to provide header updating from WCS objects.
    The parent object must have a WCS.
    """

    def wcs(self):
        raise TypeError("Classes inheriting from HeaderMixin must define a "
                        "wcs method")

    @property
    def header(self):
        header = self._nowcs_header

        wcsheader = self.wcs.to_header() if self.wcs is not None else {}

        # When preserving metadata, copy over keywords before doing the WCS
        # keyword copying, since those have specific formatting requirements
        # and will overwrite these in many cases (e.g., BMAJ)
        for key in self.meta:
            if key.upper() not in wcsheader:
                if isinstance(key, str) and len(key) <= 8:
                    try:
                        header[key.upper()] = str(self.meta[key])
                    except ValueError as ex:
                        # need a silenced-by-default warning here?
                        # log.warn("Skipped key {0} because {1}".format(key, ex))
                        pass
                elif isinstance(key, str) and len(key) > 8:
                    header['COMMENT'] = "{0}={1}".format(key, self.meta[key])

        # Preserve non-WCS information from previous header iteration
        header.update(wcsheader)
        if self.unit == u.one and 'BUNIT' in self._meta:
            # preserve the BUNIT even though it's not technically valid
            # (Jy/Beam)
            header['BUNIT'] = self._meta['BUNIT']
        else:
            header['BUNIT'] = self.unit.to_string(format='FITS')

        if 'beam' in self._meta:
            header = self._meta['beam'].attach_to_header(header)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            header.insert(2, Card(keyword='NAXIS', value=self.ndim))
            for ind,sh in enumerate(self.shape[::-1]):
                header.insert(3+ind, Card(keyword='NAXIS{0:1d}'.format(ind+1),
                                          value=sh))

        return header

    def check_jybeam_smoothing(self, raise_error_jybm=True):
        '''
        This runs for spatial resolution operations (e.g. `spatial_smooth`) and either an error or warning
        when smoothing will affect brightness in Jy/beam operations.

        This is also true for using the `with_beam` and `with_beams` methods, including 1D spectra with
        Jy/beam units.

        Parameters
        ----------
        raise_error_jybm : bool, optional
            Raises a `~spectral_cube.utils.BeamUnitsError` when True (default). When False, it triggers a
            `~spectral_cube.utils.BeamWarning`.

            .. note: This is a reminder to expose raise_error_jybm to top-level functions.

        '''

        if self.unit.is_equivalent(u.Jy / u.beam) and raise_error_jybm:
            if raise_error_jybm:
                raise BeamUnitsError("Attempting to change the spatial resolution of a cube with Jy/beam units."
                                     " To ignore this error, set `raise_error_jybm=False`.")
            else:
                warnings.warn("Changing the spatial resolution of a cube with Jy/beam units."
                              " The brightness units may be wrong!", BeamWarning)

class SpatialCoordMixinClass(object):

    @property
    def _has_wcs_celestial(self):
        return self.wcs.has_celestial

    def _raise_wcs_no_celestial(self):
        if not self._has_wcs_celestial:
            raise WCSCelestialError("WCS does not contain two spatial axes.")

    def _celestial_axes(self):
        '''
        Return the spatial axes in the data from the WCS object. The order of
        the spatial axes returned is [y, x].

        '''

        self._raise_wcs_no_celestial()

        # This works for astropy >v3
        # wcs_cel_axis = [self.wcs.world_axis_physical_types.index(axtype)
        #                 for axtype in
        #                 self.wcs.celestial.world_axis_physical_types]

        # This works for all LTS releases
        wcs_cel_axis = [ax for ax, ax_type in enumerate(self.wcs.get_axis_types()) if
                        ax_type['coordinate_type'] == 'celestial']

        # Swap to numpy ordering
        # Since we're mapping backwards to get the numpy convention, we need to
        # reverse the order at the end.
        # 0 is the y spatial axis and 1 is the x spatial axis
        np_order_cel_axis = [self.ndim - 1 - ind for ind in wcs_cel_axis][::-1]

        return np_order_cel_axis

    @cube_utils.slice_syntax
    def world(self, view):
        """
        Return a list of the world coordinates in a cube, projection, or a view
        of it.

        SpatialCoordMixinClass.world is called with *bracket notation*, like
        a NumPy array::

            c.world[0:3, :, :]

        Returns
        -------
        [v, y, x] : list of NumPy arrays
            The 3 world coordinates at each pixel in the view. For a 2D image,
            the output is ``[y, x]``.


        Examples
        --------
        Extract the first 3 velocity channels of the cube:

        >>> v, y, x = c.world[0:3]

        Extract all the world coordinates:

        >>> v, y, x = c.world[:, :, :]

        Extract every other pixel along all axes:

        >>> v, y, x = c.world[::2, ::2, ::2]

        Extract all the world coordinates for a 2D image:

        >>> y, x = c.world[:, :]

        """

        self._raise_wcs_no_celestial()

        # the next 3 lines are equivalent to (but more efficient than)
        # inds = np.indices(self._data.shape)
        # inds = [i[view] for i in inds]
        inds = np.ogrid[[slice(0, s) for s in self.shape]]
        inds = np.broadcast_arrays(*inds)
        inds = [i[view] for i in inds[::-1]]  # numpy -> wcs order

        shp = inds[0].shape
        inds = np.column_stack([i.ravel() for i in inds])
        world = self._wcs.all_pix2world(inds, 0).T

        world = [w.reshape(shp) for w in world]  # 1D->3D

        # apply units
        world = [w * u.Unit(self._wcs.wcs.cunit[i])
                 for i, w in enumerate(world)]

        # convert spectral unit if needed
        if hasattr(self, "_spectral_unit"):
            if self._spectral_unit is not None:
                specind = self.wcs.wcs.spec
                world[specind] = world[specind].to(self._spectral_unit)

        return world[::-1]  # reverse WCS -> numpy order

    def flattened_world(self, view=()):
        """
        Retrieve the world coordinates corresponding to the extracted flattened
        version of the cube
        """

        self._raise_wcs_no_celestial()

        return [wd_dim.ravel() for wd_dim in self.world[view]]

    def world_spines(self):
        """
        Returns a list of 1D arrays, for the world coordinates
        along each pixel axis.

        Raises error if this operation is ill-posed (e.g. rotated world
        coordinates, strong distortions)

        This method is not currently implemented. Use
        ``world`` instead.
        """
        raise NotImplementedError()

    @property
    def spatial_coordinate_map(self):
        view = tuple([0 for ii in range(self.ndim - 2)] + [slice(None)] * 2)
        return self.world[view][self.ndim - 2:]

    @property
    @cached
    def world_extrema(self):

        y_ax, x_ax = self._celestial_axes()

        corners = [(0, self.shape[x_ax]-1),
                   (self.shape[y_ax]-1, 0),
                   (self.shape[y_ax]-1, self.shape[x_ax]-1),
                   (0,0)]

        if len(self.shape) == 2:
            latlon_corners = [self.world[y, x] for y,x in corners]
        else:
            latlon_corners = [self.world[0, y, x][1:] for y,x in corners]

        lon = u.Quantity([x for y,x in latlon_corners])
        lat = u.Quantity([y for y,x in latlon_corners])

        _lon_min = lon.min()
        _lon_max = lon.max()
        _lat_min = lat.min()
        _lat_max = lat.max()

        return u.Quantity(((_lon_min.to(u.deg).value, _lon_max.to(u.deg).value),
                           (_lat_min.to(u.deg).value, _lat_max.to(u.deg).value)),
                          u.deg)

    @property
    @cached
    def longitude_extrema(self):
        return self.world_extrema[0]

    @property
    @cached
    def latitude_extrema(self):
        return self.world_extrema[1]


class SpectralAxisMixinClass(object):

    def _new_spectral_wcs(self, unit, velocity_convention=None,
                          rest_value=None):
        """
        Returns a new WCS with a different Spectral Axis unit

        Parameters
        ----------
        unit : :class:`~astropy.units.Unit`
            Any valid spectral unit: velocity, (wave)length, or frequency.
            Only vacuum units are supported.
        velocity_convention : 'relativistic', 'radio', or 'optical'
            The velocity convention to use for the output velocity axis.
            Required if the output type is velocity. This can be either one
            of the above strings, or an `astropy.units` equivalency.
        rest_value : :class:`~astropy.units.Quantity`
            A rest wavelength or frequency with appropriate units.  Required if
            output type is velocity.  The cube's WCS should include this
            already if the *input* type is velocity, but the WCS's rest
            wavelength/frequency can be overridden with this parameter.

            .. note: This must be the rest frequency/wavelength *in vacuum*,
                     even if your cube has air wavelength units

        """
        from .spectral_axis import (convert_spectral_axis,
                                    determine_ctype_from_vconv)

        # Allow string specification of units, for example
        if not isinstance(unit, u.Unit):
            unit = u.Unit(unit)

        # Velocity conventions: required for frq <-> velo
        # convert_spectral_axis will handle the case of no velocity
        # convention specified & one is required
        if velocity_convention in DOPPLER_CONVENTIONS:
            velocity_convention = DOPPLER_CONVENTIONS[velocity_convention]
        elif (velocity_convention is not None and
              velocity_convention not in DOPPLER_CONVENTIONS.values()):
            raise ValueError("Velocity convention must be radio, optical, "
                             "or relativistic.")

        # If rest value is specified, it must be a quantity
        if (rest_value is not None and
            (not hasattr(rest_value, 'unit') or
             not rest_value.unit.is_equivalent(u.m, u.spectral()))):
            raise ValueError("Rest value must be specified as an astropy "
                             "quantity with spectral equivalence.")

        # Shorter versions to keep lines under 80
        ctype_from_vconv = determine_ctype_from_vconv

        meta = self._meta.copy()
        if 'Original Unit' not in self._meta:
            meta['Original Unit'] = self._wcs.wcs.cunit[self._wcs.wcs.spec]
            meta['Original Type'] = self._wcs.wcs.ctype[self._wcs.wcs.spec]

        out_ctype = ctype_from_vconv(self._wcs.wcs.ctype[self._wcs.wcs.spec],
                                     unit,
                                     velocity_convention=velocity_convention)

        newwcs = convert_spectral_axis(self._wcs, unit, out_ctype,
                                       rest_value=rest_value)

        newwcs.wcs.set()
        return newwcs, meta

    @property
    def spectral_axis(self):
        # spectral objects should be forced to implement this
        raise NotImplementedError


class MaskableArrayMixinClass(object):
    """
    Mixin class for maskable arrays
    """

    def _get_filled_data(self, view=(), fill=np.nan, check_endian=False,
                         use_memmap=None):
        """
        Return the underlying data as a numpy array.
        Always returns the spectral axis as the 0th axis

        Sets masked values to *fill*
        """
        if check_endian:
            if not self._data.dtype.isnative:
                kind = str(self._data.dtype.kind)
                sz = str(self._data.dtype.itemsize)
                dt = '=' + kind + sz
                data = self._data.astype(dt)
            else:
                data = self._data
        else:
            data = self._data

        if self._mask is None:
            return data[view]

        if use_memmap is None and hasattr(self, '_is_huge'):
            use_memmap = self._is_huge

        return self._mask._filled(data=data, wcs=self._wcs, fill=fill,
                                  view=view, wcs_tolerance=self._wcs_tolerance,
                                  use_memmap=use_memmap
                                 )

    @cube_utils.slice_syntax
    def filled_data(self, view):
        """
        Return a portion of the data array, with excluded mask values
        replaced by ``fill_value``.

        Returns
        -------
        data : Quantity
            The masked data.
        """
        return u.Quantity(self._get_filled_data(view, fill=self._fill_value),
                          self.unit, copy=False)

    def filled(self, fill_value=None):
        if fill_value is not None:
            return u.Quantity(self._get_filled_data(fill=fill_value),
                              self.unit, copy=False)
        return self.filled_data[:]

    @cube_utils.slice_syntax
    def unitless_filled_data(self, view):
        """
        Return a portion of the data array, with excluded mask values
        replaced by ``fill_value``.

        Returns
        -------
        data : numpy.array
            The masked data.
        """
        return self._get_filled_data(view, fill=self._fill_value)

    @property
    def fill_value(self):
        """ The replacement value used by `~spectral_cube.base_class.MaskableArrayMixinClass.filled_data`.

        fill_value is immutable; use `~spectral_cube.base_class.MaskableArrayMixinClass.with_fill_value`
        to create a new cube with a different fill value.
        """
        return self._fill_value

    def with_fill_value(self, fill_value):
        """
        Create a new object with a different ``fill_value``.

        Notes
        -----
        This method is fast (it does not copy any data)
        """
        return self._new_thing_with(fill_value=fill_value)

    @abc.abstractmethod
    def _new_thing_with(self):
        raise NotImplementedError


class MultiBeamMixinClass(object):
    """
    A mixin class to handle multibeam objects.  To be used by
    VaryingResolutionSpectralCube's and OneDSpectrum's """

    def jtok_factors(self, equivalencies=()):
        """
        Compute an array of multiplicative factors that will convert from
        Jy/beam to K
        """

        factors = []
        for bm,frq in zip(self.beams,
                          self.with_spectral_unit(u.Hz).spectral_axis):

            # create a beam equivalency for brightness temperature
            bmequiv = bm.jtok_equiv(frq)
            factor = (u.Jy).to(u.K, equivalencies=bmequiv+list(equivalencies))
            factors.append(factor)
        factor = np.array(factors)

        return factor

    @property
    def beams(self):
        return self._beams[self.goodbeams_mask]

    @beams.setter
    def beams(self, obj):

        if not isinstance(obj, Beams):
            raise TypeError("beam must be a radio_beam.Beams object.")

        if not obj.size == self.shape[0]:
            raise ValueError("The Beams object must have the same size as the "
                             "data. Found a size of {0} and the data have a "
                             "size of {1}".format(obj.size, self.size))

        self._beams = obj

    @property
    @cached
    def pixels_per_beam(self):
        pixels_per_beam = [(beam.sr /
                (astropy.wcs.utils.proj_plane_pixel_area(self.wcs) *
                 u.deg**2)).to(u.one).value for beam in self.beams]

        return pixels_per_beam

    @property
    def unmasked_beams(self):
        return self._beams

    @property
    def goodbeams_mask(self):
        if hasattr(self, '_goodbeams_mask'):
            return self._goodbeams_mask
        else:
            return self.unmasked_beams.isfinite

    @goodbeams_mask.setter
    def goodbeams_mask(self, value):
        if value.size != self.shape[0]:
            raise ValueError("The 'good beams' mask must have the same size "
                             "as the cube's spectral dimension")

        self._goodbeams_mask = value

    def identify_bad_beams(self, threshold, reference_beam=None,
                           criteria=['sr','major','minor'],
                           mid_value=np.nanmedian):
        """
        Mask out any layers in the cube that have beams that differ from the
        central value of the beam by more than the specified threshold.

        Parameters
        ----------
        threshold : float
            Fractional threshold
        reference_beam : Beam
            A beam to use as the reference.  If unspecified, ``mid_value`` will
            be used to select a middle beam
        criteria : list
            A list of criteria to compare.  Can include
            'sr','major','minor','pa' or any subset of those.
        mid_value : function
            The function used to determine the 'mid' value to compare to.  This
            will identify the middle-valued beam area/major/minor/pa.

        Returns
        -------
        includemask : np.array
            A boolean array where ``True`` indicates the good beams
        """

        includemask = np.ones(self.unmasked_beams.size, dtype='bool')

        all_criteria = {'sr','major','minor','pa'}
        if not set.issubset(set(criteria), set(all_criteria)):
            raise ValueError("Criteria must be one of the allowed options: "
                             "{0}".format(all_criteria))

        props = {prop: u.Quantity([getattr(beam, prop) for beam in self.unmasked_beams])
                 for prop in all_criteria}

        if reference_beam is None:
            reference_beam = Beam(major=mid_value(props['major']),
                                  minor=mid_value(props['minor']),
                                  pa=mid_value(props['pa'])
                                 )

        for prop in criteria:
            val = props[prop]
            mid = getattr(reference_beam, prop)

            diff = np.abs((val-mid)/mid)

            assert diff.shape == includemask.shape

            includemask[diff > threshold] = False

        return includemask

    def average_beams(self, threshold, mask='compute', warn=False):
        """
        Average the beams.  Note that this operation only makes sense in
        limited contexts!  Generally one would want to convolve all the beams
        to a common shape, but this method is meant to handle the "simple" case
        when all your beams are the same to within some small factor and can
        therefore be arithmetically averaged.

        Parameters
        ----------
        threshold : float
            The fractional difference between beam major, minor, and pa to
            permit
        mask : 'compute', None, or boolean array
            The mask to apply to the beams.  Useful for excluding bad channels
            and edge beams.
        warn : bool
            Warn if successful?

        Returns
        -------
        new_beam : radio_beam.Beam
            A new radio beam object that is the average of the unmasked beams
        """

        use_dask = isinstance(self._data, da.Array)

        if mask == 'compute':
            if use_dask:
                # If we are dealing with dask arrays, we compute the beam
                # mask once and for all since it is used multiple times in its
                # entirety in the remainder of this method.
                beam_mask = da.any(da.logical_and(self._mask_include,
                                                  self.goodbeams_mask[:, None, None]),
                                   axis=(1, 2))
                # da.any appears to return an object dtype instead of a bool
                beam_mask = self._compute(beam_mask).astype('bool')
            elif self.mask is not None:
                beam_mask = np.any(np.logical_and(self.mask.include(),
                                                  self.goodbeams_mask[:, None, None]),
                                   axis=(1, 2))
            else:
                beam_mask = self.goodbeams_mask
        else:
            if mask.ndim > 1:
                beam_mask = np.logical_and(mask, self.goodbeams_mask[:, None, None])
            else:
                beam_mask = np.logical_and(mask, self.goodbeams_mask)

        if not any(beam_mask):
            raise ValueError("All beams were excluded using threshold {threshold}"
                             .format(threshold=threshold))

        # use private _beams here because the public one excludes the bad beams
        # by default
        new_beam = self._beams.average_beam(includemask=beam_mask)

        if np.isnan(new_beam):
            raise ValueError("Beam was not finite after averaging.  "
                             "This either indicates that there was a problem "
                             "with the include mask, one of the beam's values, "
                             "or a bug.")

        self._check_beam_areas(threshold, mean_beam=new_beam, mask=beam_mask)
        if warn:
            warnings.warn("Arithmetic beam averaging is being performed.  This is "
                          "not a mathematically robust operation, but is being "
                          "permitted because the beams differ by "
                          "<{0}".format(threshold),
                          BeamAverageWarning
                         )
        return new_beam


    def _handle_beam_areas_wrapper(self, function, beam_threshold=None):
        """
        Wrapper: if the function takes "axis" and is operating over axis 0 (the
        spectral axis), check that the beam threshold is not exceeded before
        performing the operation

        Also, if the operation *is* valid, average the beam appropriately to
        get the output
        """
        # deferred import to avoid a circular import problem
        from .lower_dimensional_structures import LowerDimensionalObject

        if beam_threshold is None:
            beam_threshold = self.beam_threshold

        def newfunc(*args, **kwargs):
            """ Wrapper function around the standard operations to handle beams
            when creating projections """

            # check that the spectral axis is being operated over.  If it is,
            # we need to average beams
            # moments are a special case b/c they default to axis=0
            need_to_handle_beams = (('axis' in kwargs and
                                     ((kwargs['axis']==0) or
                                      (hasattr(kwargs['axis'], '__len__') and
                                       0 in kwargs['axis']))) or
                                    ('axis' not in kwargs and 'moment' in
                                     function.__name__))

            if need_to_handle_beams:
                # do this check *first* so we don't do an expensive operation
                # and crash afterward
                avg_beam = self.average_beams(beam_threshold, warn=True)

            result = function(*args, **kwargs)

            if not isinstance(result, LowerDimensionalObject):
                # numpy arrays are sometimes returned; these have no metadata
                return result

            elif need_to_handle_beams:
                result.meta['beam'] = avg_beam
                result._beam = avg_beam

            return result

        return newfunc

    def _check_beam_areas(self, threshold, mean_beam, mask=None):
        """
        Check that the beam areas are the same to within some threshold
        """

        if mask is not None:
            assert len(mask) == len(self.unmasked_beams)
            mask = np.array(mask, dtype='bool')
        else:
            mask = np.ones(len(self.unmasked_beams), dtype='bool')

        qtys = dict(sr=self.unmasked_beams.sr,
                    major=self.unmasked_beams.major.to(u.deg),
                    minor=self.unmasked_beams.minor.to(u.deg),
                    # position angles are not really comparable
                    #pa=u.Quantity([bm.pa for bm in self.unmasked_beams], u.deg),
                   )

        errormessage = ""

        for (qtyname, qty) in (qtys.items()):
            minv = qty[mask].min()
            maxv = qty[mask].max()
            mn = getattr(mean_beam, qtyname)
            maxdiff = (np.max(np.abs(u.Quantity((maxv-mn, minv-mn))))/mn).decompose()

            if isinstance(threshold, dict):
                th = threshold[qtyname]
            else:
                th = threshold

            if maxdiff > th:
                errormessage += ("Beam {2}s differ by up to {0}x, which is greater"
                                 " than the threshold {1}\n".format(maxdiff,
                                                                    threshold,
                                                                    qtyname
                                                                   ))
        if errormessage != "":
            raise ValueError(errormessage)

    def mask_out_bad_beams(self, threshold, reference_beam=None,
                           criteria=['sr','major','minor'],
                           mid_value=np.nanmedian):
        """
        See `identify_bad_beams`.  This function returns a masked cube

        Returns
        -------
        newcube : VaryingResolutionSpectralCube
            The cube with bad beams masked out
        """

        goodbeams = self.identify_bad_beams(threshold=threshold,
                                            reference_beam=reference_beam,
                                            criteria=criteria,
                                            mid_value=mid_value)

        includemask = BooleanArrayMask(goodbeams[:, None, None],
                                       self._wcs,
                                       shape=self._data.shape)

        use_dask = isinstance(self._data, da.Array)
        if use_dask:
            newmask = da.logical_and(self._mask_include,
                                     includemask)
        elif self.mask is None:
            newmask = includemask
        else:
            newmask = np.bitwise_and(self.mask, includemask)

        return self._new_thing_with(mask=newmask,
                                    beam_threshold=threshold,
                                    goodbeams_mask=np.bitwise_and(self.goodbeams_mask, goodbeams),
                                   )

    def with_beams(self, beams, goodbeams_mask=None, raise_error_jybm=True):
        '''
        Attach a new beams object to the VaryingResolutionSpectralCube.

        Parameters
        ----------
        beams : `~radio_beam.Beams`
            A new beams object.
        '''

        # Catch cases with units in Jy/beam where new beams will alter the units.
        self.check_jybeam_smoothing(raise_error_jybm=raise_error_jybm)

        meta = self.meta.copy()
        meta['beams'] = beams

        return self._new_thing_with(beams=beams, meta=meta)

    @abc.abstractmethod
    def _new_thing_with(self):
        # since the above two methods require this method, it's an ABC of this
        # mixin as well
        raise NotImplementedError



class BeamMixinClass(object):
    """
    Functionality for objects with a single beam.

    Specific objects (cubes, LDOs) still need to define their own ``with_beam``
    methods.
    """

    @property
    def beam(self):
        if self._beam is None:
            raise NoBeamError("No beam is defined for this SpectralCube or the"
                              " beam information could not be parsed from the"
                              " header. A `~radio_beam.Beam` object can be"
                              " added using `cube.with_beam`.")

        return self._beam

    @beam.setter
    def beam(self, obj):

        if not isinstance(obj, Beam) and obj is not None:
            raise TypeError("beam must be a radio_beam.Beam object.")

        self._beam = obj


    @property
    @cached
    def pixels_per_beam(self):
        return (self.beam.sr /
                (astropy.wcs.utils.proj_plane_pixel_area(self.wcs) *
                 u.deg**2)).to(u.one).value
