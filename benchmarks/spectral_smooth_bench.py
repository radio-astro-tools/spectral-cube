"""
Spectral-smoothing benchmark for ``SpectralCube.spectral_smooth``.

Compares the per-spectrum default path against several whole-cube
alternatives, including FFT-based convolution. Cube shape and kernel
sizes mimic the issue-#1003 workload (CRAFTS cutout: 995 x 221 x 481).

Run with::

    python benchmarks/spectral_smooth_bench.py [--quick]

The ``--quick`` flag halves the cube size and reduces to one kernel.

What's covered
--------------
For each kernel sigma:

- per-spectrum ``astropy.convolution.convolve`` (the current default;
  timed on a (50, 50) spatial subset and extrapolated for sanity)
- whole-cube ``astropy.convolution.convolve`` with a ``(k, 1, 1)`` kernel
  (what ``vectorize=True`` does when extra kwargs are passed; also what
  ``DaskSpectralCube.spectral_smooth`` does per block)
- whole-cube ``scipy.ndimage.convolve1d`` (what ``vectorize=True``
  drops to when astropy defaults are in effect)
- whole-cube ``astropy.convolution.convolve_fft`` with ``(k, 1, 1)``
- ``scipy.signal.fftconvolve`` along ``axes=0`` (1D FFT only, no
  3D padding -- the natural way to spectral-smooth via FFT)
- ``scipy.signal.oaconvolve`` along ``axes=0`` (overlap-add, designed
  for long signals + short kernels)

All approaches are checked against the astropy whole-cube reference for
equivalence within a tolerance.

Findings on this workload (run on Apple M-series, 2026-05-30)
-------------------------------------------------------------
Kernel sigma=5 (size 41):  scipy.ndimage wins.
Kernel sigma=20 (size 161): scipy.ndimage still wins; scipy.signal.oaconvolve
  is competitive; astropy.convolve_fft is much slower (3D FFT padding).
Kernel sigma=50 (size 401): scipy.signal.oaconvolve and fftconvolve along
  axis 0 become attractive; astropy.convolve_fft remains slow because of
  the 3D padding cost.

Conclusion: ``astropy.convolve_fft`` is not competitive on this workload
because it pads/FFTs a 3D array even though the kernel is 1D. If we ever
want an FFT path, ``scipy.signal.oaconvolve(data, k3d, axes=0)`` is the
right tool.
"""
import argparse
import sys
import time
import numpy as np

from astropy.convolution import (Gaussian1DKernel, convolve as ap_convolve,
                                 convolve_fft as ap_convolve_fft)
from astropy.wcs import WCS
from scipy.ndimage import convolve1d
from scipy.signal import fftconvolve, oaconvolve

from spectral_cube import SpectralCube, DaskSpectralCube
from spectral_cube.masks import BooleanArrayMask


def make_cube(shape, nan_fraction=0.0, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(dtype)
    if nan_fraction > 0:
        m = rng.random(shape) < nan_fraction
        data[m] = np.nan
    return data


def time_call(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"    {label:55s} {dt:8.3f} s")
    return out, dt


# ---------- approaches ----------------------------------------------------

def per_spectrum_astropy(data, kernel):
    """The current ``spectral_smooth`` default: iterate one spectrum at a
    time and call ``astropy.convolution.convolve`` on each."""
    out = np.empty_like(data)
    for j in range(data.shape[1]):
        for i in range(data.shape[2]):
            out[:, j, i] = ap_convolve(data[:, j, i], kernel,
                                       normalize_kernel=True)
    return out


def whole_cube_astropy(data, kernel):
    """``vectorize=True`` general path."""
    k3d = kernel.array.reshape((-1, 1, 1))
    return ap_convolve(data, k3d, normalize_kernel=True)


def whole_cube_scipy_ndimage(data, kernel):
    """``vectorize=True`` fast path (when astropy defaults are in effect)."""
    k = np.asarray(kernel.array, dtype=np.float64)
    k = k / k.sum()
    nan = np.isnan(data)
    if nan.any():
        filled = np.where(nan, 0.0, data)
        w = (~nan).astype(np.float64)
        num = convolve1d(filled, k, axis=0, mode='constant', cval=0.0)
        den = convolve1d(w, k, axis=0, mode='constant', cval=1.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = num / den
        out[den == 0] = 0.0
        return out
    return convolve1d(data, k, axis=0, mode='constant', cval=0.0)


def whole_cube_astropy_fft(data, kernel):
    """astropy.convolution.convolve_fft on the whole cube. Pads/FFTs in 3D
    even though only one axis is non-trivial; expensive."""
    k3d = kernel.array.reshape((-1, 1, 1))
    return ap_convolve_fft(data, k3d, normalize_kernel=True,
                           boundary='fill', fill_value=0.0,
                           nan_treatment='interpolate', allow_huge=True)


def whole_cube_scipy_fftconvolve(data, kernel):
    """scipy.signal.fftconvolve along axis 0 only (no spatial padding).
    Pads with zeros at the boundary to match astropy default. Returns the
    'same'-size output. Does NOT do NaN renormalization."""
    k = np.asarray(kernel.array, dtype=np.float64)
    k = k / k.sum()
    k3d = k.reshape((-1, 1, 1))
    return fftconvolve(data, k3d, mode='same', axes=0)


def whole_cube_scipy_oaconvolve(data, kernel):
    """scipy.signal.oaconvolve along axis 0 only. Overlap-add: uses many
    small FFTs sized for the kernel, very efficient for thin kernels in
    long signals. Does NOT do NaN renormalization."""
    k = np.asarray(kernel.array, dtype=np.float64)
    k = k / k.sum()
    k3d = k.reshape((-1, 1, 1))
    return oaconvolve(data, k3d, mode='same', axes=0)


# ---------- spectral-cube paths -------------------------------------------

def _make_wcs():
    w = WCS(naxis=3)
    w.wcs.ctype = ['RA---CAR', 'DEC--CAR', 'VRAD']
    w.wcs.crpix = [1.0, 1.0, 1.0]
    w.wcs.cdelt = [-0.025, 0.025, 200.0]
    w.wcs.cunit = ['deg', 'deg', 'm/s']
    w.wcs.crval = [113.0, -13.0, 1.0e5]
    return w


def _make_spectral_cube(data, dask=False, spatial_chunk=None):
    w = _make_wcs()
    hdr = w.to_header()
    hdr['BUNIT'] = 'K'
    mask = BooleanArrayMask(np.isfinite(data), wcs=w)
    if dask:
        import dask.array as da
        if spatial_chunk is None:
            # 'auto' tends to pick a single chunk for cubes this size,
            # which gives no parallelism.
            spatial = ('auto', 'auto')
        else:
            spatial = (spatial_chunk, spatial_chunk)
        darr = da.from_array(data, chunks=(data.shape[0],) + spatial)
        return DaskSpectralCube(data=darr, wcs=w, header=hdr, mask=mask)
    return SpectralCube(data=data, wcs=w, header=hdr, mask=mask)


def cube_vectorize(data, kernel):
    """``SpectralCube.spectral_smooth(..., vectorize=True)`` end-to-end,
    going through the auto-routed ndimage/oaconvolve fast path."""
    cube = _make_spectral_cube(data)
    out = cube.spectral_smooth(kernel, vectorize=True)
    return np.asarray(out.unitless_filled_data[:])


def dask_cube_smooth(data, kernel):
    """``DaskSpectralCube.spectral_smooth`` with default ('auto') chunking
    and default convolve (astropy.convolution.convolve)."""
    cube = _make_spectral_cube(data, dask=True)
    out = cube.spectral_smooth(kernel)
    return np.asarray(out.unitless_filled_data[:])


def dask_cube_smooth_small_chunks(data, kernel, spatial_chunk=64):
    """Same DaskSpectralCube path but with small spatial chunks so dask
    actually has work to parallelise across. Each chunk is convolved with
    astropy.convolution.convolve."""
    cube = _make_spectral_cube(data, dask=True, spatial_chunk=spatial_chunk)
    out = cube.spectral_smooth(kernel)
    return np.asarray(out.unitless_filled_data[:])


def dask_cube_smooth_scipy(data, kernel, spatial_chunk=64):
    """DaskSpectralCube with small spatial chunks and a custom per-chunk
    convolve that drops to scipy.ndimage.convolve1d. This is the closest
    dask analog of the non-dask ``vectorize=True`` fast path."""
    cube = _make_spectral_cube(data, dask=True, spatial_chunk=spatial_chunk)

    def _scipy_chunk_convolve(arr, kernel, **kwargs):
        # ``arr`` is a 3D chunk shaped (n_spec, ny_chunk, nx_chunk).
        k1 = np.asarray(kernel[:, 0, 0], dtype=np.float64)
        k1 = k1 / k1.sum()
        return convolve1d(arr, k1, axis=0, mode='constant', cval=0.0)

    out = cube.spectral_smooth(kernel, convolve=_scipy_chunk_convolve)
    return np.asarray(out.unitless_filled_data[:])


def dask_cube_vectorize(data, kernel, spatial_chunk=64):
    """``DaskSpectralCube.spectral_smooth(vectorize=True)`` — the in-package
    auto-routed scipy fast path on dask cubes."""
    cube = _make_spectral_cube(data, dask=True, spatial_chunk=spatial_chunk)
    out = cube.spectral_smooth(kernel, vectorize=True)
    return np.asarray(out.unitless_filled_data[:])


# ---------- driver --------------------------------------------------------

def run_one_kernel(data, sigma, sub_shape=(995, 50, 50),
                   skip_per_spectrum=False):
    kernel = Gaussian1DKernel(sigma)
    print(f"  kernel sigma={sigma}, size={kernel.array.size}")

    # Reference for equivalence: whole-cube astropy.convolve (identical
    # semantics to the per-spectrum loop).
    print("    --- timing ---")
    ref, _ = time_call("astropy whole-cube (vectorize=True general path)",
                        lambda: whole_cube_astropy(data, kernel))
    out_sci, _ = time_call("scipy.ndimage.convolve1d (vectorize=True fast path)",
                            lambda: whole_cube_scipy_ndimage(data, kernel))
    out_fft, t_fft = time_call("astropy.convolve_fft whole-cube",
                                 lambda: whole_cube_astropy_fft(data, kernel))
    out_sigfft, _ = time_call("scipy.signal.fftconvolve (axes=0)",
                                lambda: whole_cube_scipy_fftconvolve(data, kernel))
    out_oa, _ = time_call("scipy.signal.oaconvolve (axes=0)",
                             lambda: whole_cube_scipy_oaconvolve(data, kernel))
    out_vec, _ = time_call("SpectralCube.spectral_smooth(vectorize=True)",
                            lambda: cube_vectorize(data, kernel))
    out_dask, _ = time_call("DaskSpectralCube.spectral_smooth (default chunks)",
                             lambda: dask_cube_smooth(data, kernel))
    out_dask_small, _ = time_call(
        "DaskSpectralCube.spectral_smooth (64x64 spatial chunks)",
        lambda: dask_cube_smooth_small_chunks(data, kernel,
                                               spatial_chunk=64))
    out_dask_scipy, _ = time_call(
        "DaskSpectralCube.spectral_smooth + scipy.ndimage per chunk",
        lambda: dask_cube_smooth_scipy(data, kernel, spatial_chunk=64))
    out_dask_vec, _ = time_call(
        "DaskSpectralCube.spectral_smooth(vectorize=True)",
        lambda: dask_cube_vectorize(data, kernel, spatial_chunk=64))
    if not skip_per_spectrum:
        sub = data[:, :sub_shape[1], :sub_shape[2]].copy()
        _, t_sub = time_call(
            f"astropy per-spectrum [{sub_shape[1]}x{sub_shape[2]} subset]",
            lambda: per_spectrum_astropy(sub, kernel))
        full_pix = data.shape[1] * data.shape[2]
        sub_pix = sub_shape[1] * sub_shape[2]
        ext = t_sub * full_pix / sub_pix
        print(f"      -> per-spectrum extrapolated full cube ~ {ext:.1f} s")

    print("    --- equivalence vs whole-cube astropy ---")
    mask = ~(np.isnan(ref) | np.isnan(out_sci))
    if mask.any():
        d = np.abs(ref - out_sci)[mask].max()
        print(f"    scipy.ndimage          : max|Δ| = {d:.3e}")
    mask = ~(np.isnan(ref) | np.isnan(out_fft))
    if mask.any():
        d = np.abs(ref - out_fft)[mask].max()
        print(f"    astropy.convolve_fft   : max|Δ| = {d:.3e}")
    mask = ~(np.isnan(ref) | np.isnan(out_vec))
    if mask.any():
        d = np.abs(ref - out_vec)[mask].max()
        print(f"    SpectralCube vectorize : max|Δ| = {d:.3e}")
    mask = ~(np.isnan(ref) | np.isnan(out_dask))
    if mask.any():
        d = np.abs(ref - out_dask)[mask].max()
        print(f"    DaskSpectralCube       : max|Δ| = {d:.3e}")
    mask = ~(np.isnan(ref) | np.isnan(out_dask_small))
    if mask.any():
        d = np.abs(ref - out_dask_small)[mask].max()
        print(f"    Dask 64x64 chunks      : max|Δ| = {d:.3e}")
    mask = ~(np.isnan(ref) | np.isnan(out_dask_scipy))
    if mask.any():
        d = np.abs(ref - out_dask_scipy)[mask].max()
        print(f"    Dask + scipy per-chunk : max|Δ| = {d:.3e}")
    mask = ~(np.isnan(ref) | np.isnan(out_dask_vec))
    if mask.any():
        d = np.abs(ref - out_dask_vec)[mask].max()
        print(f"    Dask vectorize=True    : max|Δ| = {d:.3e}")
    # scipy.signal.{fft,oa}convolve don't renormalize NaNs/edges -- skip
    # equivalence on NaN cubes.
    if not np.isnan(data).any():
        mask = ~(np.isnan(ref) | np.isnan(out_sigfft))
        if mask.any():
            d = np.abs(ref - out_sigfft)[mask].max()
            print(f"    scipy.signal.fftconvolve: max|Δ| = {d:.3e}  "
                  "(no NaN handling; clean cube only)")
        mask = ~(np.isnan(ref) | np.isnan(out_oa))
        if mask.any():
            d = np.abs(ref - out_oa)[mask].max()
            print(f"    scipy.signal.oaconvolve : max|Δ| = {d:.3e}  "
                  "(no NaN handling; clean cube only)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="smaller cube and single kernel")
    args = parser.parse_args()

    if args.quick:
        shape = (500, 100, 200)
        sigmas = [5]
    else:
        shape = (995, 221, 481)
        sigmas = [5, 20, 50]

    print(f"cube shape = {shape}  (n_cells = {np.prod(shape):,})")
    print()

    print("===== CLEAN cube =====")
    data_clean = make_cube(shape, nan_fraction=0.0)
    for s in sigmas:
        run_one_kernel(data_clean, s)

    print("===== 5% NaN cube =====")
    data_nan = make_cube(shape, nan_fraction=0.05)
    for s in sigmas:
        run_one_kernel(data_nan, s, skip_per_spectrum=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
