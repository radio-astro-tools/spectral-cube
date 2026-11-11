import numpy as np
from astropy.wcs import WCS
import warnings
from astropy import units as u
from astropy import log
from astropy.wcs import (InconsistentAxisTypesError,
                         NonseparableSubimageCoordinateSystemError)
from astropy.utils.shapes import simplify_basic_index

from .utils import WCSWarning

bad_spectypes_mapping = {'VELOCITY':'VELO',
                         'WAVELENG':'WAVE',
                         }


def _strip_celestial_projection(wcs, context):
    """
    Return a copy of the WCS with the projection code removed from the
    celestial axes, which turns them into linear axes so that one of them
    can be dropped.
    """
    wcs = wcs.deepcopy()
    prj = wcs.wcs.ctype[wcs.wcs.lng].split("-")[-1]
    for ind in wcs.wcs.lng, wcs.wcs.lat:
        wcs.wcs.ctype[ind] = wcs.wcs.ctype[ind].split("-")[0]
    warnings.warn("Slicing across a celestial axis results "
                  "in an invalid WCS, so the celestial "
                  "projection ({0}) is being removed.  "
                  "{1}".format(prj, context),
                  WCSWarning)
    wcs.wcs.set_pv([])
    wcs.wcs.set()
    return wcs


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
    # Negative indices count from the last axis
    dropax = range(wcs.wcs.naxis)[dropax]

    try:
        return wcs.dropaxis(dropax)
    except (InconsistentAxisTypesError,
            NonseparableSubimageCoordinateSystemError):
        pass

    if dropax in (wcs.wcs.lng, wcs.wcs.lat):
        wcs = _strip_celestial_projection(
            wcs, "The WCS index being dropped was {0}.".format(dropax))
    else:
        wcs = wcs.deepcopy()

    # Remove any coupling between the dropped axis and the remaining axes so
    # that the remaining axes keep their diagonal and mutual terms.
    if wcs.wcs.has_cd():
        matrix = wcs.wcs.cd.copy()
    else:
        matrix = wcs.wcs.get_pc().copy()
    keep = np.arange(wcs.wcs.naxis) != dropax
    matrix[dropax, keep] = 0
    matrix[keep, dropax] = 0
    if wcs.wcs.has_cd():
        wcs.wcs.cd = matrix
    else:
        wcs.wcs.pc = matrix
    wcs.wcs.set()

    return wcs.dropaxis(dropax)


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

    if shape is not None:
        view = simplify_basic_index(view, shape=tuple(map(int, shape)))

    if not all([isinstance(x, slice) for x in view]):
        if drop_degenerate:
            keeps = [mywcs.naxis-ii
                     for ii,ind in enumerate(view)
                     if isinstance(ind, slice)]
            keeps.sort()
            try:
                mywcs = mywcs.sub(keeps)
            except InconsistentAxisTypesError as ex:
                # make a copy of the WCS because we need to modify it inplace
                wcscp = mywcs.deepcopy()
                for ct in wcscp.celestial.wcs.ctype:
                    match = list(wcscp.wcs.ctype).index(ct)
                    prj = wcscp.wcs.ctype[match].split("-")[-1]
                    wcscp.wcs.ctype[match] = wcscp.wcs.ctype[match].split("-")[0]
                    warnings.warn("Slicing across a celestial axis results "
                                  "in an invalid WCS, so the celestial "
                                  "projection ({0}) is being removed.  "
                                  "The view used was {1}."
                                  .format(prj, view),
                                  WCSWarning)
                mywcs = wcscp.sub(keeps)
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
            crval = mywcs.sub([wcs_index+1]).wcs_pix2world([refpix-1], 0)[0][0]
            crpix = 1
            cdelt = mywcs.wcs.cdelt[wcs_index]

            wcs_new.wcs.crpix[wcs_index] = crpix
            wcs_new.wcs.crval[wcs_index] = crval
            wcs_new.wcs.cdelt[wcs_index] = -cdelt

        elif iview.start is not None:

            if iview.step not in (None, 1):
                crpix = mywcs.wcs.crpix[wcs_index]
                cdelt = mywcs.wcs.cdelt[wcs_index]

                # the logic is very annoying: the blc of the first pixel
                # is at 0.5, so that value must be subtracted to get into
                # numpy-compatible coordinates, then added back afterward
                crp = ((crpix - iview.start - 0.5)/iview.step + 0.5)
                # SIMPLE TEST:
                # view(0, None, 1) w/crpix = 1
                # crp = 1
                # view(0, None, 2) w/crpix = 1
                # crp = 0.75
                # view(0, None, 4) w/crpix = 1
                # crp = 0.625
                # view(2, None, 1) w/crpix = 1
                # crp = -1
                # view(2, None, 2) w/crpix = 1
                # crp = -0.25
                # view(2, None, 4) w/crpix = 1
                # crp = 0.125

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
            elif isinstance(c1[1], float):
                try:
                    if exact:
                        assert c1[1] == c2[1]
                    else:
                        np.testing.assert_almost_equal(c1[1], c2[1], decimal=decimal)
                except AssertionError:
                    if key in ('RESTFRQ','RESTWAV'):
                        warnings.warn("{0} is not equal in WCS; ignoring ".format(key)+
                                      "under the assumption that you want to"
                                      " compare velocity cubes.", WCSWarning)
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
                warnings.warn("WCS2 is missing card {0}".format(key), WCSWarning)
            elif key not in ignore_keywords:
                OK = False

    # Check that there aren't any cards in header 2 that were missing from
    # header 1
    for c2 in h2.cards:
        key = c2[0]
        if key not in matched:
            if warn_missing:
                warnings.warn("WCS1 is missing card {0}".format(key), WCSWarning)
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
               'PC0{ii}_0{jj}', 'CD{ii}_{jj}', 'CROTA{ii}', 'PC{ii}_{jj}',
               'PC{ii:03d}{jj:03d}', 'PV0{ii}_0{jj}', 'PV{ii}_{jj}'):
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


def is_pixel_axis_to_wcs_correlated(mywcs, axis):
    """
    Check if the chosen pixel axis correlates to other WCS axes. This tests
    whether the pixel axis is correlated only to 1 WCS axis and can be
    considered independent of the others.
    """

    axis_corr_matrix = mywcs.axis_correlation_matrix

    # Map from numpy axis to WCS axis
    wcs_axis = mywcs.world_n_dim - (axis + 1)

    # Grab the row along the given spatial axis. This slice is along the WCS axes
    wcs_axis_correlations = axis_corr_matrix[:, wcs_axis]

    # The image axis should always be correlated to at least 1 WCS axis.
    # i.e., the diagonal term is one in the matrix. Correlations with other axes will give
    # a sum > 1
    if wcs_axis_correlations.sum() > 1:
        return True

    return False


def find_spatial_pixel_index(cube, xlo, xhi, ylo, yhi):
    '''
    Given low and high cuts, return the pixel coordinates for a rectangular
    region in the given cube or spatial projection. lo and hi inputs can be
    given in pixels, "min"/"max", or in world coordinates.

    When spatial WCS dimensions are given as an `~astropy.units.Quantity`,
    the spatial coordinates of the 'lo' and 'hi' corners are solved together.
    This minimizes WCS variations due to the sky curvature when slicing from
    a large (>1 deg) image.


    Parameters
    ----------
    cube : :class:`~SpectralCube` or spatial :class:`~Projection`
        A spectral-cube or projection/slice with spatial dimensions.
    [xy]lo/[xy]hi : int or :class:`~astropy.units.Quantity` or ``min``/``max``
        The endpoints to extract.  If given as a ``Quantity``, will be
        interpreted as World coordinates.  If given as a ``string`` or
        ``int``, will be interpreted as pixel coordinates.

    Returns
    -------
    limit_dict : dict
        Pixel coordinates of [xy]lo/[xy]hi in the given ``cube``.

    '''

    ndim = cube.ndim

    for val in (xlo,ylo,xhi,yhi):
        if hasattr(val, 'unit') and not val.unit.is_equivalent(u.degree):
            raise u.UnitsError("The X and Y slices must be specified in "
                                "degree-equivalent units.")

    limit_dict = {}

    # Match corners. If one uses a WCS coord, set 'min'/'max'
    # To the lat or long extrema.
    # We only care about matching spatial corners.
    xlo_unit = hasattr(xlo, 'unit')
    ylo_unit = hasattr(ylo, 'unit')

    # Do min/max switching if the WCS grid increases/decreases
    # with the pixel grid.
    ymin = min if cube.wcs.wcs.cdelt[1] > 0 else max
    xmin = min if cube.wcs.wcs.cdelt[0] > 0 else max
    ymax = max if cube.wcs.wcs.cdelt[1] > 0 else min
    xmax = max if cube.wcs.wcs.cdelt[0] > 0 else min

    if not any([xlo_unit, ylo_unit]):
        limit_dict['xlo'] = 0 if xlo == 'min' else xlo
        limit_dict['ylo'] = 0 if ylo == 'min' else ylo
    else:
        if xlo_unit:
            limit_dict['xlo'] = xlo
            limit_dict['ylo'] = ymin(cube.latitude_extrema) if ylo == 'min' else ylo
        if ylo_unit:
            limit_dict['ylo'] = ylo
            limit_dict['xlo'] = xmin(cube.longitude_extrema) if xlo == 'min' else xlo

    xhi_unit = hasattr(xhi, 'unit')
    yhi_unit = hasattr(yhi, 'unit')

    if not any([xhi_unit, yhi_unit]):

        # For 3D cube
        if ndim == 3:
            limit_dict['xhi'] = cube.shape[2] if xhi == 'max' else xhi
            limit_dict['yhi'] = cube.shape[1] if yhi == 'max' else yhi
        # For 2D spatial projection/slice
        else:
            limit_dict['xhi'] = cube.shape[1] if xhi == 'max' else xhi
            limit_dict['yhi'] = cube.shape[0] if yhi == 'max' else yhi
    else:
        if xhi_unit:
            limit_dict['xhi'] = xhi
            limit_dict['yhi'] = ymax(cube.latitude_extrema) if yhi == 'max' else yhi
        if yhi_unit:
            limit_dict['yhi'] = yhi
            limit_dict['xhi'] = xmax(cube.longitude_extrema) if xhi == 'max' else xhi

    # list to track which entries had units
    united = []

    # Solve the spatial axes together.
    # There's 3 options:
    # (1) If both pixel units, do nothing
    # (2) If both WCS units, use world_to_array_index_values
    # (3) If mixed, minimize the distance between the spatial position grids
    #     for the cube to find the closest spatial pixel.

    for corn in ['lo', 'hi']:
        grids = {}

        # Check if either were given as a WCS value with a unit
        x_hasunit = hasattr(limit_dict['x'+corn], 'unit')
        y_hasunit = hasattr(limit_dict['y'+corn], 'unit')

        # print(limit_dict['x'+corn], limit_dict['y'+corn])
        # print(x_hasunit, y_hasunit)

        # (1) If both pixel units, we keep in pixel units.
        if not any([x_hasunit, y_hasunit]):
            continue

        # (2) If both WCS units, use world_to_array_index_values
        elif all([x_hasunit, y_hasunit]):

            corn_arr = np.array([limit_dict['x'+corn].value,
                                 limit_dict['y'+corn].value])

            xmin, ymin = cube.wcs.celestial.world_to_array_index_values(corn_arr.reshape((1, 2)))[0]

            limit_dict['y' + corn] = ymin
            limit_dict['x' + corn] = xmin

            if corn == 'hi':
                united.append('y' + corn)
                united.append('x' + corn)


        # (3) If mixed, minimize the distance between the spatial position grids
        #     for the cube to find the closest spatial pixel, limited to the 1 pixel
        #     value that is given.
        else:

            # We change the dimensions being sliced depending on whether the
            # x or y dim is given in pixel units.
            # This allows for a 1D minimization instead of needing both spatial axes.

            if x_hasunit:
                pixval = limit_dict['y' + corn]
                lim = 'x' + corn
                slicedim = 0
            else:
                pixval = limit_dict['x' + corn]
                lim = 'y' + corn
                slicedim = 1

            if corn == 'lo':
                slice_pixdim = slice(pixval, pixval+1)
            else:
                slice_pixdim = slice(pixval-1, pixval)

            limval = limit_dict[lim]
            if hasattr(limval, 'unit'):
                united.append(lim)

                sl = [slice(None)]
                sl.insert(slicedim, slice_pixdim)

                if ndim == 3:
                    sl.insert(0, slice(0, 1))


                sl = tuple(sl)

                if slicedim == 0:
                    spine = cube.world[sl][2 if ndim == 3 else 1]
                else:
                    spine = cube.world[sl][1 if ndim == 3 else 0]

                val = np.argmin(np.abs(limval-spine))
                if limval > spine.max() or limval < spine.min():
                    log.warning("The limit {0} is out of bounds."
                                "  Using min/max instead.".format(lim))

                limit_dict[lim] = val

    # Correct ordering (this shouldn't be necessary but do a quick check)
    for xx in 'yx':
        hi,lo = limit_dict[xx+'hi'], limit_dict[xx+'lo']
        if hi < lo:
            # must have high > low
            limit_dict[xx+'hi'], limit_dict[xx+'lo'] = lo, hi

        if xx+'hi' in united:
            # End-inclusive indexing: need to add one for the high slice
            # Only do this for converted values, not for pixel values
            # (i.e., if the xlo/ylo/zlo value had units)
            limit_dict[xx+'hi'] += 1

    return limit_dict
