import numpy as np

from astropy import units as u
from astropy.extern.six.moves import zip
from astropy.extern.six.moves import range as xrange
from astropy.wcs import WCS
from astropy.utils.console import ProgressBar

from .cube_utils import _map_context
from .lower_dimensional_structures import OneDSpectrum


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
    slices = [slice(None) if ii == axis else None for ii in range(x.ndim)]
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
                       [chunk * i for i in xrange(int(num_items / chunk))])
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
                               " spectral unit ().".format(spec_unit))

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

    pix_shifts = vdiff_sign * ((velocity_surface.to(vel_unit) -
                                v0.to(vel_unit)) / vdiff).value[xy_posns]

    # May a header copy so we can start altering
    new_header = cube[:, 0, 0].header.copy()

    if pad_edges:
        # Enables padding the whole cube such that no spectrum will wrap around
        # This is critical if a low-SB component is far off of the bright
        # component that the velocity surface is derived from.

        # Find max +/- pixel shifts, rounding up to the nearest integer
        max_pos_shift = np.ceil(np.nanmax(pix_shifts)).astype(int)
        max_neg_shift = np.ceil(np.nanmin(pix_shifts)).astype(int)

        # The total pixel size of the new spectral axis
        num_vel_pix = cube.spectral_axis.size + max_pos_shift - max_neg_shift
        new_header['NAXIS1'] = num_vel_pix

        # Adjust CRPIX in header
        new_header['CRPIX1'] += max_pos_shift

        pad_size = (max_pos_shift, -max_neg_shift)

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

    stacked = stack_function(all_shifted_spectra, axis=0)

    stack_spec = \
        OneDSpectrum(stacked, unit=cube.unit, wcs=WCS(new_header),
                     header=new_header,
                     meta=cube.meta, spectral_unit=vel_unit,
                     beams=cube.beams if hasattr(cube, "beams") else None)

    return stack_spec


def stack_cube(cube, linelist, vmin, vmax, average=np.nanmean):
    """
    Create a stacked cube by averaging on a common velocity grid.

    Parameters
    ----------
    cube : SpectralCube
        The cube
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
    """

    line_cube = cube.with_spectral_unit(u.km/u.s,
                                        velocity_convention='radio',
                                        rest_value=linelist[0])
    reference_cube = line_cube.spectral_slab(vmin, vmax)

    cutout_cubes = [reference_cube[:].value]

    for restval in linelist[1:]:
        line_cube = cube.with_spectral_unit(u.km/u.s,
                                            velocity_convention='radio',
                                            rest_value=restval)
        line_cutout = line_cube.spectral_slab(vmin, vmax)

        regridded = line_cutout.spectral_interpolate(reference_cube.spectral_axis)

        cutout_cubes.append(regridded[:].value)

    stacked_cube = average(cutout_cubes, axis=0)

    hdu = reference_cube.hdu

    hdu.data = stacked_cube

    return hdu
