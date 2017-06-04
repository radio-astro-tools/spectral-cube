from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.wcs import WCS, InconsistentAxisTypesError
import warnings
from astropy import units as u
from astropy import log

wcs_parameters_to_preserve = ['cel_offset', 'dateavg', 'dateobs', 'equinox',
                              'latpole', 'lonpole', 'mjdavg', 'mjdobs', 'name',
                              'obsgeo', 'phi0', 'radesys', 'restfrq',
                              'restwav', 'specsys', 'ssysobs', 'ssyssrc',
                              'theta0', 'velangl', 'velosys', 'zsource']
# not writable:
# 'lat', 'lng', 'lattyp', 'lngtyp',

bad_spectypes_mapping = {'VELOCITY':'VELO',
                         'WAVELENG':'WAVE',
                         }

class WCSWrapper(WCS):
    """
    Wrapper of WCS to deal with some of the special cases we face within
    spectral_cube
    """
    @property
    def has_celestial(self):
        if hasattr(self, '_has_celestial'):
            return self._has_celestial
        try:
            return self.celestial.naxis == 2
        except InconsistentAxisTypesError:
            return False

    @has_celestial.setter
    def has_celestial(self, value):
        if value is not False:
            warnings.warn("_has_celestial is being set to {0}, "
                          "which may not be what you want."
                          .format(value))
        self._has_celestial = value

    def wcs_pix2world(self, pixels, reference):
        if ((pixels.shape[1] < self.naxis and
            hasattr(self, 'active_dimensions') and
             len(self.active_dimensions) < self.naxis)):
            pixels = np.asarray(pixels)
            pixels = np.c_[pixels, np.zeros(pixels.shape[0])]
            result = super(WCSWrapper, self).wcs_pix2world(pixels, reference)
            return result[:, :len(self.active_dimensions) - self.naxis]
        else:
            return super(WCSWrapper, self).wcs_pix2world(pixels, reference)

def drop_axis(wcs, dropax):
    """
    Drop the ax on axis dropax

    Remove an axis from the WCS
    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS with naxis to be chopped to naxis-1
    dropax: int
        The index of the WCS to drop, counting from 0 (i.e., python convention,
        not FITS convention)
    """
    inds = list(range(wcs.wcs.naxis))
    inds.pop(dropax)
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def add_stokes_axis_to_wcs(wcs, add_before_ind):
    """
    Add a new Stokes axis that is uncorrelated with any other axes

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to add to
    add_before_ind: int
        Index of the WCS to insert the new Stokes axis in front of.
        To add at the end, do add_before_ind = wcs.wcs.naxis
    """

    naxin = wcs.wcs.naxis
    naxout = naxin + 1

    inds = list(range(naxout))
    inds.pop(add_before_ind)
    inds = np.array(inds)

    outwcs = WCS(naxis=naxout)
    for par in wcs_parameters_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs, par))

    pc = np.zeros([naxout, naxout])
    pc[inds[:, np.newaxis], inds[np.newaxis, :]] = wcs.wcs.get_pc()
    pc[add_before_ind, add_before_ind] = 1

    def append_to_posn(val, posn, lst):
        """ insert a value at index into a list """
        return list(lst)[:posn] + [val] + list(lst)[posn:]

    outwcs.wcs.crpix = append_to_posn(1, add_before_ind, wcs.wcs.crpix)
    outwcs.wcs.cdelt = append_to_posn(1, add_before_ind, wcs.wcs.get_cdelt())
    outwcs.wcs.crval = append_to_posn(1, add_before_ind, wcs.wcs.crval)
    outwcs.wcs.cunit = append_to_posn("", add_before_ind, wcs.wcs.cunit)
    outwcs.wcs.ctype = append_to_posn("STOKES", add_before_ind, wcs.wcs.ctype)
    outwcs.wcs.cname = append_to_posn("STOKES", add_before_ind, wcs.wcs.cname)
    outwcs.wcs.pc = pc

    return outwcs


def wcs_swapaxes(wcs, ax0, ax1):
    """
    Swap axes in a WCS

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to have its axes swapped
    ax0: int
    ax1: int
        The indices of the WCS to be swapped, counting from 0 (i.e., python
        convention, not FITS convention)
    """
    inds = list(range(wcs.wcs.naxis))
    inds[ax0], inds[ax1] = inds[ax1], inds[ax0]
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def reindex_wcs(wcs, inds):
    """
    Re-index a WCS given indices.  The number of axes may be reduced.

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to be manipulated
    inds: np.array(dtype='int')
        The indices of the array to keep in the output.
        e.g. swapaxes: [0,2,1,3]
        dropaxes: [0,1,3]
    """

    if not isinstance(inds, np.ndarray):
        raise TypeError("Indices must be an ndarray")

    if inds.dtype.kind != 'i':
        raise TypeError('Indices must be integers')

    outwcs = WCS(naxis=len(inds))
    for par in wcs_parameters_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs, par))

    cdelt = wcs.wcs.get_cdelt()
    pc = wcs.wcs.get_pc()

    outwcs.wcs.crpix = wcs.wcs.crpix[inds]
    outwcs.wcs.cdelt = cdelt[inds]
    outwcs.wcs.crval = wcs.wcs.crval[inds]
    outwcs.wcs.cunit = [wcs.wcs.cunit[i] for i in inds]
    outwcs.wcs.ctype = [wcs.wcs.ctype[i] for i in inds]
    outwcs.wcs.cname = [wcs.wcs.cname[i] for i in inds]
    outwcs.wcs.pc = pc[inds[:, None], inds[None, :]]

    pv_cards = []
    for i, j in enumerate(inds):
        for k, m, v in wcs.wcs.get_pv():
            if k == j:
                pv_cards.append((i, m, v))
    outwcs.wcs.set_pv(pv_cards)

    ps_cards = []
    for i, j in enumerate(inds):
        for k, m, v in wcs.wcs.get_ps():
            if k == j:
                ps_cards.append((i, m, v))
    outwcs.wcs.set_ps(ps_cards)

    return outwcs


def axis_names(wcs):
    """
    Extract world names for each coordinate axis

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to extract names from

    Returns
    -------
    A tuple of names along each axis
    """
    names = list(wcs.wcs.cname)
    types = wcs.wcs.ctype
    for i in range(len(names)):
        if len(names[i]) > 0:
            continue
        names[i] = types[i].split('-')[0]
    return names


def slice_wcs(mywcs, view, shape=None, numpy_order=True,
              drop_degenerate=False):
    """
    Slice a WCS instance using a Numpy slice. The order of the slice should
    be reversed (as for the data) compared to the natural WCS order.

    Parameters
    ----------
    view : tuple
        A tuple containing the same number of slices as the WCS system.
        The ``step`` method, the third argument to a slice, is not
        presently supported.
    numpy_order : bool
        Use numpy order, i.e. slice the WCS so that an identical slice
        applied to a numpy array will slice the array and WCS in the same
        way. If set to `False`, the WCS will be sliced in FITS order,
        meaning the first slice will be applied to the *last* numpy index
        but the *first* WCS axis.
    drop_degenerate : bool
        Drop axes that are size-1, i.e., any that have an integer index as part
        of their view?  Otherwise, an Exception will be raised.

    Returns
    -------
    wcs_new : `~astropy.wcs.WCS`
        A new resampled WCS axis
    """
    if hasattr(view, '__len__') and len(view) > mywcs.wcs.naxis:
        raise ValueError("Must have # of slices <= # of WCS axes")
    elif not hasattr(view, '__len__'): # view MUST be an iterable
        view = [view]

    if not all([isinstance(x, slice) for x in view]):
        if drop_degenerate:
            keeps = [mywcs.naxis-ii
                     for ii,ind in enumerate(view)
                     if isinstance(ind, slice)]
            mywcs = mywcs.sub(keeps)
            view = [x for x in view if isinstance(x, slice)]
        else:
            raise ValueError("Cannot downsample a WCS with indexing.  Use "
                             "wcs.sub or wcs.dropaxis if you want to remove "
                             "axes.")

    wcs_new = mywcs.deepcopy()
    for i, iview in enumerate(view):
        if iview.step is not None and iview.start is None:
            # Slice from "None" is equivalent to slice from 0 (but one
            # might want to downsample, so allow slices with
            # None,None,step or None,stop,step)
            iview = slice(0, iview.stop, iview.step)

        if numpy_order:
            wcs_index = mywcs.wcs.naxis - 1 - i
        else:
            wcs_index = i

        if iview.step is not None and iview.step < 0:
            if iview.step != -1:
                raise NotImplementedError("Haven't dealt with resampling & reversing.")
            # reverse indexing requires the use of shape
            if shape is None:
                raise ValueError("Cannot reverse-index a WCS without "
                                 "specifying a shape.")
            if iview.stop is not None:
                refpix = iview.stop
            else:
                refpix = shape[i]
            # this will raise an inconsistent axis type error if slicing over
            # celestial axes is attempted
            # wcs_index+1 is required because sub([0]) = sub([all])
            crval = mywcs.sub([wcs_index+1]).wcs_pix2world([refpix], 0)[0]
            crpix = 1
            wcs_new.wcs.crpix[wcs_index] = crpix
            wcs_new.wcs.crval[wcs_index] = crval

        elif iview.start is not None:

            if iview.step not in (None, 1):
                crpix = mywcs.wcs.crpix[wcs_index]
                cdelt = mywcs.wcs.cdelt[wcs_index]
                # equivalently (keep this comment so you can compare eqns):
                # wcs_new.wcs.crpix[wcs_index] =
                # (crpix - iview.start)*iview.step + 0.5 - iview.step/2.
                crp = ((crpix - iview.start - 1.)/iview.step +
                       0.5 + 1./iview.step/2.)
                wcs_new.wcs.crpix[wcs_index] = crp
                wcs_new.wcs.cdelt[wcs_index] = cdelt * iview.step
            else:
                wcs_new.wcs.crpix[wcs_index] -= iview.start

    # Without this, may cause a regression of #234
    wcs_new.wcs.set()

    return wcs_new

def check_equality(wcs1, wcs2, warn_missing=False,
                   ignore_keywords=['MJD-OBS', 'VELOSYS'],
                   wcs_tolerance=0.0):
    """
    Check if two WCSs are equal

    Parameters
    ----------
    wcs1, wcs2: `astropy.wcs.WCS`
        The WCSs
    warn_missing: bool
        Issue warnings if one header is missing a keyword that the other has?
    ignore_keywords: list of str
        Keywords that are stored as part of the WCS but do not define part of
        the coordinate system and therefore can be safely ignored.
    wcs_tolerance : float
        The decimal level to check for equality.
        For example, 1e-2 would have 0.001 and 0.002 as equal, but 1e-3 would
        have them as inequal
    """
    # TODO: use this to replace the rest of the check_equality code
    #return wcs1.wcs.compare(wcs2.wcs, cmp=wcs.WCSCOMPARE_ANCILLARY,
    #                        tolerance=tolerance)
    #Until we've switched to the wcs.compare approach, we need to have
    #np.testing.assert_almost_equal work
    if wcs_tolerance == 0:
        exact = True
    else:
        exact = False
        # np.testing.assert_almost_equal wants an integer
        # e.g., for 0.0001, the integer is 4
        decimal = int(np.ceil(-np.log10(wcs_tolerance)))


    # naive version:
    # return str(wcs1.to_header()) != str(wcs2.to_header())

    h1 = wcs1.to_header()
    h2 = wcs2.to_header()

    # Default to headers equal; everything below changes to false if there are
    # any inequalities
    OK = True
    # to keep track of keywords in both
    matched = []

    for c1 in h1.cards:
        key = c1[0]
        if key in h2:
            matched.append(key)
            c2 = h2.cards[key]
            # special check for units: "m/s" = "m s-1"
            if 'UNIT' in key:
                u1 = u.Unit(c1[1])
                u2 = u.Unit(c2[1])
                if u1 != u2:
                    if key in ignore_keywords:
                        log.debug("IGNORED Header 1, {0}: {1} != {2}".format(key,u1,u2))
                    else:
                        OK = False
                        log.debug("Header 1, {0}: {1} != {2}".format(key,u1,u2))
            elif isinstance(c1[1], (float, np.float)):
                try:
                    if exact:
                        assert c1[1] == c2[1]
                    else:
                        np.testing.assert_almost_equal(c1[1], c2[1], decimal=decimal)
                except AssertionError:
                    if key in ('RESTFRQ','RESTWAV'):
                        warnings.warn("{0} is not equal in WCS; ignoring ".format(key)+
                                      "under the assumption that you want to"
                                      " compare velocity cubes.")
                        continue
                    if key in ignore_keywords:
                        log.debug("IGNORED Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                    else:
                        log.debug("Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                        OK = False
            elif c1[1] != c2[1]:
                if key in ignore_keywords:
                    log.debug("IGNORED Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                else:
                    log.debug("Header 1, {0}: {1} != {2}".format(key,c1[1],c2[1]))
                    OK = False
        else:
            if warn_missing:
                warnings.warn("WCS2 is missing card {0}".format(key))
            elif key not in ignore_keywords:
                OK = False

    # Check that there aren't any cards in header 2 that were missing from
    # header 1
    for c2 in h2.cards:
        key = c2[0]
        if key not in matched:
            if warn_missing:
                warnings.warn("WCS1 is missing card {0}".format(key))
            else:
                OK = False

    return OK

def strip_wcs_from_header(header):
    """
    Given a header with WCS information, remove ALL WCS information from that
    header
    """

    hwcs = WCS(header)
    wcsh = hwcs.to_header()

    keys_to_keep = [k for k in header
                    if (k and k not in wcsh and 'NAXIS' not in k)]

    newheader = header.copy()

    # Strip blanks first.  They appear to cause serious problems, like not
    # deleting things they should!
    if '' in newheader:
        del newheader['']

    for kw in list(newheader.keys()):
        if kw not in keys_to_keep:
            del newheader[kw]

    for kw in ('CRPIX{ii}', 'CRVAL{ii}', 'CDELT{ii}', 'CUNIT{ii}', 'CTYPE{ii}',
               'PC0{ii}_0{jj}', 'CD{ii}_{jj}', 'CROTA{ii}'):
        for ii in range(5):
            for jj in range(5):
                k = kw.format(ii=ii,jj=jj)
                if k in newheader.keys():
                    del newheader[k]


    return newheader

def diagonal_wcs_to_cdelt(mywcs):
    """
    If a WCS has only diagonal pixel scale matrix elements (which are composed
    from cdelt*pc), use them to reform the wcs as a CDELT-style wcs with no pc
    or cd elements
    """
    offdiag = ~np.eye(mywcs.pixel_scale_matrix.shape[0], dtype='bool')
    if not any(mywcs.pixel_scale_matrix[offdiag]):
        cdelt = mywcs.pixel_scale_matrix.diagonal()
        del mywcs.wcs.pc
        del mywcs.wcs.cd
        mywcs.wcs.cdelt = cdelt
    return mywcs

def drop_axis_by_slicing(mywcs, shape, dropped_axis,
                         dropped_axis_slice_position='middle',
                         dropped_axis_cdelt='same',
                         convert_misaligned_to_offset=True,
                        ):
    """
    Parameters
    ----------
    dropped_axis_slice_position : 'middle', 'start', 'end'
        If an axis is being dropped, where should the WCS say the
        projection is?  It can be at the start, middle, or end of the
        axis.
    dropped_axis_cdelt : 'same', 'full_range', or value
        If an axis is being dropped, what should the new CDELT be?  For an
        integral, for example, one might want the value to be the full
        range.  For a slice, it should stay the same.  For something like
        min or max, it might be zero.
    convert_misaligned_to_offset : bool
        If the axes are misaligned, it is not possible to "drop" an axis.
        In this case, a generic "offset axis" will be returned.
    """
    ndim = len(shape)

    if mywcs.get_axis_types()[dropped_axis]['coordinate_type'] == 'celestial':
        dropping_celestial = True
    else:
        dropping_celestial = False

    if dropped_axis_slice_position == 'middle':
        dropped_axis_slice_position = shape[dropped_axis]//2
    elif dropped_axis_slice_position == 'start':
        dropped_axis_slice_position = 0
    elif dropped_axis_slice_position == 'end':
        dropped_axis_slice_position = shape[dropped_axis]

    dropax_slice = slice(dropped_axis_slice_position,
                         dropped_axis_slice_position+1)

    view = [slice(None) if ax!=dropped_axis else dropax_slice
            for ax in range(ndim)]

    crpix_new = [0 if ax!=dropped_axis else dropped_axis_slice_position
                 for ax in range(ndim)]
    new_crval = mywcs.wcs_pix2world([crpix_new], 0)[0, dropped_axis]

    result = slice_wcs(mywcs, view, shape=shape)

    result.wcs.crval[dropped_axis] = new_crval
    result.wcs.crpix[dropped_axis] = 1

    if dropped_axis_cdelt == 'same':
        dropped_axis_cdelt = mywcs.wcs.cdelt[dropped_axis]
    elif dropped_axis_cdelt == 'full_range':
        ref_pixels = np.array(
            [(0,0) if ax!=dropped_axis else (0, shape[dropped_axis])
             for ax in range(ndim)])
        refvals = mywcs.wcs_pix2world(ref_pixels.T, 0)[:, dropped_axis]
        dropped_axis_cdelt = refvals[1]-refvals[0]

    result.wcs.cdelt[dropped_axis] = dropped_axis_cdelt

    new_inds = np.array([ii for ii in range(ndim) if ii != dropped_axis] +
                        [dropped_axis])
    result = reindex_wcs(result, new_inds)

    if dropping_celestial:
        new_result = WCSWrapper()
        new_result.wcs = result.wcs
        new_result.has_celestial = False
        new_result.active_dimensions = list(range(ndim-1))
        result = new_result

    return result
