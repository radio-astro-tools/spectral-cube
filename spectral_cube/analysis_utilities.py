import numpy as np

from astropy import units as u
from six.moves import zip, range
from astropy.wcs import WCS
from astropy.utils.console import ProgressBar
from astropy import log
import warnings

from .utils import BadVelocitiesWarning
from .cube_utils import _map_context
from .lower_dimensional_structures import VaryingResolutionOneDSpectrum, OneDSpectrum
from .spectral_cube import VaryingResolutionSpectralCube, SpectralCube


def fourier_shift(x, shift, axis=0, add_pad=False, pad_size=None):
    '''
    Shift a spectrum in the Fourier plane.

    Parameters
    ----------
    x : np.ndarray
        Array to be shifted
    shift : int or float
        Number of pixels to shift.
    axis : int, optional
        Axis to shift along.
    pad_size : int, optional
        Pad the array before shifting.

    Returns
    -------
    x2 : np.ndarray
        Shifted array.

    '''
    nanmask = ~np.isfinite(x)

    # If all NaNs, there is nothing to shift
    # But only if there is no added padding. Otherwise we need to pad
    if nanmask.all() and not add_pad:
        return x

    nonan = x.copy()

    shift_mask = False
    if nanmask.any():
        nonan[nanmask] = 0.0
        shift_mask = True

    # Optionally pad the edges
    if add_pad:
        if pad_size is None:
            # Pad by the size of the shift
            pad = np.ceil(shift).astype(int)

            # Determine edge to pad whether it is a positive or negative shift
            pad_size = (pad, 0) if shift > 0 else (0, pad)
        else:
            assert len(pad_size)

        pad_nonan = np.pad(nonan, pad_size, mode='constant',
                           constant_values=(0))
        if shift_mask:
            pad_mask = np.pad(nanmask, pad_size, mode='constant',
                              constant_values=(0))
    else:
        pad_nonan = nonan
        pad_mask = nanmask

    # Check if there are all NaNs before shifting
    if nanmask.all():
        return np.array([np.NaN] * pad_mask.size)

    nonan_shift = _fourier_shifter(pad_nonan, shift, axis)
    if shift_mask:
        mask_shift = _fourier_shifter(pad_mask, shift, axis) > 0.5
        nonan_shift[mask_shift] = np.NaN

    return nonan_shift


def _fourier_shifter(x, shift, axis):
    '''
    Helper function for `~fourier_shift`.
    '''
    ftx = np.fft.fft(x, axis=axis)
    m = np.fft.fftfreq(x.shape[axis])
    # m_shape = [1] * x.ndim
    # m_shape[axis] = m.shape[0]
    # m = m.reshape(m_shape)
    slices = tuple([slice(None) if ii == axis else None for ii in range(x.ndim)])
    m = m[slices]
    phase = np.exp(-2 * np.pi * m * 1j * shift)
    x2 = np.real(np.fft.ifft(ftx * phase, axis=axis))
    return x2


def get_chunks(num_items, chunk):
    '''
    Parameters
    ----------
    num_items : int
        Number of total items.
    chunk : int
        Size of chunks

    Returns
    -------
    chunks : list of np.ndarray
        List of channels in chunks of the given size.
    '''
    items = np.arange(num_items)

    if num_items == chunk:
        return [items]

    chunks = \
        np.array_split(items,
                       [chunk * i for i in range(int(num_items / chunk))])
    if chunks[-1].size == 0:
        # Last one is empty
        chunks = chunks[:-1]
    if chunks[0].size == 0:
        # First one is empty
        chunks = chunks[1:]

    return chunks


def _spectrum_shifter(inputs):
    spec, shift, add_pad, pad_size = inputs
    return fourier_shift(spec, shift, add_pad=add_pad, pad_size=pad_size)


def stack_spectra(cube, velocity_surface, v0=None,
                  stack_function=np.nanmean,
                  xy_posns=None, num_cores=1,
                  chunk_size=-1,
                  progressbar=False, pad_edges=True,
                  vdiff_tol=0.01):
    '''
    Shift spectra in a cube according to a given velocity surface (peak
    velocity, centroid, rotation model, etc.).

    Parameters
    ----------
    cube : SpectralCube
        The cube
    velocity_field : Quantity
        A Quantity array with m/s or equivalent units
    stack_function : function
        A function that can operate over a list of numpy arrays (and accepts
        ``axis=0``) to combine the spectra.  `numpy.nanmean` is the default,
        though one might consider `numpy.mean` or `numpy.median` as other
        options.
    xy_posns : list, optional
        List the spatial positions to include in the stack. For example,
        if the data is masked by some criterion, the valid points can be given
        as `xy_posns = np.where(mask)`.
    num_cores : int, optional
        Choose number of cores to run on. Defaults to 1.
    chunk_size : int, optional
        To limit memory usage, the shuffling of spectra can be done in chunks.
        Chunk size sets the number of spectra that, if memory-mapping is used,
        is the number of spectra loaded into memory. Defaults to -1, which is
        all spectra.
    progressbar : bool, optional
        Print progress through every chunk iteration.
    pad_edges : bool, optional
        Pad the edges of the shuffled spectra to stop data from rolling over.
        Default is True. The rolling over occurs since the FFT treats the
        boundary as periodic. This should only be disabled if you know that
        the velocity range exceeds the range that a spectrum has to be
        shuffled to reach `v0`.
    vdiff_tol : float, optional
        Allowed tolerance for changes in the spectral axis spacing. Default
        is 0.01, or 1%.

    Returns
    -------
    stack_spec : OneDSpectrum
        The stacked spectrum.
    '''

    if not np.isfinite(velocity_surface).any():
        raise ValueError("velocity_surface contains no finite values.")

    vshape = velocity_surface.shape
    cshape = cube.shape[1:]
    if not (vshape == cshape):
        raise ValueError("Velocity surface map does not match cube spatial "
                         "dimensions.")

    if xy_posns is None:
        # Only compute where a shift can be found
        xy_posns = np.where(np.isfinite(velocity_surface))

    if v0 is None:
        # Set to the mean velocity of the cube if not given.
        v0 = cube.spectral_axis.mean()
    else:
        if not isinstance(v0, u.Quantity):
            raise u.UnitsError("v0 must be a quantity.")
        spec_unit = cube.spectral_axis.unit
        if not v0.unit.is_equivalent(spec_unit):
            raise u.UnitsError("v0 must have units equivalent to the cube's"
                               f" spectral unit {spec_unit}.")

        v0 = v0.to(spec_unit)

        if v0 < cube.spectral_axis.min() or v0 > cube.spectral_axis.max():
            raise ValueError("v0 must be within the range of the spectral "
                             "axis of the cube.")

    # Calculate the pixel shifts that will be applied.
    spec_size = np.diff(cube.spectral_axis[:2])[0]
    # Assign the correct +/- for pixel shifts based on whether the spectral
    # axis is increasing (-1) or decreasing (+1)
    vdiff_sign = -1. if spec_size.value > 0. else 1.
    vdiff = np.abs(spec_size)
    vel_unit = vdiff.unit

    # Check to make sure vdiff doesn't change more than the allowed tolerance
    # over the spectral axis
    vdiff2 = np.abs(np.diff(cube.spectral_axis[-2:])[0])
    if not np.isclose(vdiff2.value, vdiff.value, rtol=vdiff_tol):
        raise ValueError("Cannot shift spectra on a non-linear axes")

    vmax = cube.spectral_axis.to(vel_unit).max()
    vmin = cube.spectral_axis.to(vel_unit).min()

    if ((np.any(velocity_surface > vmax) or
         np.any(velocity_surface < vmin))):
        warnings.warn("Some velocities are outside the allowed range and will be "
                      "masked out.", BadVelocitiesWarning)
        # issue 580/1 note: numpy <=1.16 will strip units from velocity, >=
        # 1.17 will not
        masked_velocities = np.where(
            (velocity_surface < vmax) &
            (velocity_surface > vmin),
            velocity_surface.value, np.nan)
        velocity_surface = u.Quantity(masked_velocities, velocity_surface.unit)

    pix_shifts = vdiff_sign * ((velocity_surface.to(vel_unit) -
                                v0.to(vel_unit)) / vdiff).value[xy_posns]

    # Make a header copy so we can start altering
    new_header = cube[:, 0, 0].header.copy()

    if pad_edges:
        # Enables padding the whole cube such that no spectrum will wrap around
        # This is critical if a low-SB component is far off of the bright
        # component that the velocity surface is derived from.

        # Find max +/- pixel shifts, rounding up to the nearest integer
        max_pos_shift = np.ceil(np.nanmax(pix_shifts)).astype(int)
        max_neg_shift = np.ceil(np.nanmin(pix_shifts)).astype(int)
        if max_neg_shift > 0:
            # if there are no negative shifts, we can ignore them and just
            # use the positive shift
            max_neg_shift = 0
        if max_pos_shift < 0:
            # same for positive
            max_pos_shift = 0

        # The total pixel size of the new spectral axis
        num_vel_pix = cube.spectral_axis.size + max_pos_shift - max_neg_shift
        new_header['NAXIS1'] = num_vel_pix

        # Adjust CRPIX in header
        new_header['CRPIX1'] += -max_neg_shift

        pad_size = (-max_neg_shift, max_pos_shift)

    else:
        pad_size = None

    all_shifted_spectra = []

    if chunk_size == -1:
        chunk_size = len(xy_posns[0])

    # Create chunks of spectra for read-out.
    chunks = get_chunks(len(xy_posns[0]), chunk_size)
    if progressbar:
        iterat = ProgressBar(chunks)
    else:
        iterat = chunks

    for i, chunk in enumerate(iterat):

        gen = ((cube.filled_data[:, y, x].value, shift, pad_edges, pad_size)
               for y, x, shift in
               zip(xy_posns[0][chunk], xy_posns[1][chunk], pix_shifts[chunk]))

        with _map_context(num_cores) as map:

            shifted_spectra = map(_spectrum_shifter, gen)

            all_shifted_spectra.extend([out for out in shifted_spectra])

    shifted_spectra_array = np.array(all_shifted_spectra)
    assert shifted_spectra_array.ndim == 2

    stacked = stack_function(shifted_spectra_array, axis=0)

    if hasattr(cube, 'beams'):
        stack_spec = VaryingResolutionOneDSpectrum(stacked, unit=cube.unit,
                                                   wcs=WCS(new_header),
                                                   header=new_header,
                                                   meta=cube.meta,
                                                   spectral_unit=vel_unit,
                                                   beams=cube.beams)
    else:
        stack_spec = OneDSpectrum(stacked, unit=cube.unit, wcs=WCS(new_header),
                                  header=new_header, meta=cube.meta,
                                  spectral_unit=vel_unit, beam=cube.beam)

    return stack_spec


def stack_cube(cube, linelist, vmin, vmax, average=np.nanmean,
               convolve_beam=None, return_hdu=False,
               return_cutouts=False):
    """
    Create a stacked cube by averaging on a common velocity grid.

    If the input cubes have varying resolution, this will trigger potentially
    expensive convolutions.

    Parameters
    ----------
    cube : SpectralCube
        The cube (or a list of cubes)
    linelist : list of Quantities
        An iterable of Quantities representing line rest frequencies
    vmin / vmax : Quantity
        Velocity-equivalent quantities specifying the velocity range to average
        over
    average : function
        A function that can operate over a list of numpy arrays (and accepts
        ``axis=0``) to average the spectra.  `numpy.nanmean` is the default,
        though one might consider `numpy.mean` or `numpy.median` as other
        options.
    convolve_beam : None
        If the cube is a VaryingResolutionSpectralCube, a convolution beam is
        required to put the cube onto a common grid prior to spectral
        interpolation.
    return_hdu : bool
        Return an HDU instead of a spectral-cube
    return_cutouts : bool
        Also return the individual cube cutouts?

    Returns
    =======
    cube : SpectralCube
        The SpectralCube object containing the reprojected cube.  Its header
        will be based on the first cube but will have no reference frequency.
        Its spectral axis will be in velocity units.
    cutout_cubes : list
        A list of cube cutouts projected into the same space (optional; see
        ``return_cutouts``)
    """

    if isinstance(cube, list):
        cubes = cube
        cube = cubes[0]
        for cb in cubes[1:]:
            if cb.shape[1:] != cube.shape[1:]:
                raise ValueError("If you pass multiple cubes, they must have the "
                                 "same spatial shape.")
        if convolve_beam is None and (any(hasattr(cb, 'beams') for cb in cubes) or
                                      not all([cb.beam == cube.beam for cb in cubes[1:]])):
            raise ValueError("If the cubes have different resolution, `convolve_beam` must be specified.")
    else:
        cubes = [cube]

    slabs = []
    included_lines = []

    # loop over linelist first to keep the cutouts in the same order as the
    # input line frequencies
    for restval in linelist:
        for cube in cubes:
            line_cube = cube.with_spectral_unit(u.km/u.s,
                                                velocity_convention='radio',
                                                rest_value=restval)
            line_cutout = line_cube.spectral_slab(vmin, vmax)
            if line_cutout.shape[0] <= 1:
                log.debug(f"Skipped line {restval} for cube {cube} because it resulted"
                          "in a size-1 spectral axis")
                continue
            else:
                included_lines.append(restval)

            assert line_cutout.shape[0] > 1

            if isinstance(line_cutout, VaryingResolutionSpectralCube):
                if convolve_beam is None:
                    raise ValueError("If any of the input cubes have varyin resolution, "
                                     "a target `common_beam` must be specified.")
                line_cutout = line_cutout.convolve_to(convolve_beam)

            assert not isinstance(line_cutout, VaryingResolutionSpectralCube)
            slabs.append(line_cutout)

    reference_cube = slabs[0]
    cutout_cubes = [reference_cube.filled_data[:].value]
    for slab in slabs[1:]:
        regridded = slab.spectral_interpolate(reference_cube.spectral_axis)
        cutout_cubes.append(regridded.filled_data[:].value)

    stacked_cube = average(cutout_cubes, axis=0)

    ww = reference_cube.wcs.copy()
    # set restfreq to zero: it is not defined any more.
    ww.wcs.restfrq = 0.0
    meta = reference_cube.meta
    meta.update({'stacked_lines': included_lines})
    result_cube = SpectralCube(data=stacked_cube,
                               wcs=ww,
                               meta=meta,
                               header=reference_cube.header)

    if return_hdu:
        retval = result_cube.hdu
    else:
        retval = result_cube

    if return_cutouts:
        return retval, cutout_cubes
    else:
        return retval
