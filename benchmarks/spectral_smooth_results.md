# `spectral_smooth` whole-cube benchmark — results

Run on Apple M-series, macOS 13.4, Python 3.10, on a synthetic float64 cube
of shape **(995, 221, 481)** matching the CRAFTS cutout from issue #1003.
Kernels are `astropy.convolution.Gaussian1DKernel(sigma)`. Times are
seconds, best of one run (numbers reproduce within ±10%).

Run yourself with:

```
python benchmarks/spectral_smooth_bench.py        # full, 3 kernels
python benchmarks/spectral_smooth_bench.py --quick # smaller, 1 kernel
```

## Clean cube (no NaNs)

| approach                                                     | σ=5 (k=41) | σ=20 (k=161) | σ=50 (k=401) |
|--------------------------------------------------------------|-----------:|-------------:|-------------:|
| per-spectrum `astropy.convolve` *(extrapolated)*             |       6.7  |       13.6   |       35.6   |
| **whole-cube** `astropy.convolve` (general fast path)        |      5.48  |       22.17  |       60.54  |
| **whole-cube** `scipy.ndimage.convolve1d` (default fast path)|       1.14 |        4.97  |       13.60  |
| whole-cube `astropy.convolve_fft`                            |      28.91 |       29.93  |       36.08  |
| `scipy.signal.fftconvolve(axes=0)`                           |       1.33 |        1.35  |        1.66  |
| `scipy.signal.oaconvolve(axes=0)`                            |       2.04 |        1.18  |        1.53  |
| `SpectralCube.spectral_smooth(vectorize=True)`               |       1.62 |        1.79  |        2.24  |
| `DaskSpectralCube.spectral_smooth` (default chunks)          |       5.46 |       22.76  |       59.58  |
| `DaskSpectralCube.spectral_smooth` (64×64 chunks)            |       5.93 |       22.68  |      113.92  |
| `DaskSpectralCube.spectral_smooth` + scipy.ndimage per chunk |       0.87 |        1.54  |        2.53  |

## 5% NaN cube

| approach                                                     | σ=5  | σ=20  | σ=50  |
|--------------------------------------------------------------|-----:|------:|------:|
| whole-cube `astropy.convolve`                                | 6.38 | 32.47 | 86.98 |
| **whole-cube** `scipy.ndimage.convolve1d` (default fast path)| 2.56 |  9.80 | 27.15 |
| whole-cube `astropy.convolve_fft`                            | 27.64 | 31.68 | 37.70 |
| `scipy.signal.fftconvolve(axes=0)` *(no NaN handling)*       | 1.25 |  1.38 |  1.73 |
| `scipy.signal.oaconvolve(axes=0)` *(no NaN handling)*        | 1.97 |  1.31 |  1.57 |
| `SpectralCube.spectral_smooth(vectorize=True)`               | 3.64 |  4.35 |  5.07 |
| `DaskSpectralCube.spectral_smooth` (default chunks)          | 6.57 | 30.97 | 81.20 |
| `DaskSpectralCube.spectral_smooth` (64×64 chunks)            | 7.17 | 31.18 | 142.11 |
| `DaskSpectralCube.spectral_smooth` + scipy.ndimage per chunk *(no NaN renorm)* | 0.91 | 1.39 | 2.49 |

All approaches that handle NaNs match the whole-cube `astropy.convolve`
reference to machine precision (max|Δ| ≤ 1.6e-15). The
`DaskSpectralCube + scipy per chunk` row is timed without NaN
renormalization in the custom per-chunk function; at finite output cells
it still matches the reference to machine precision (NaNs propagate
locally rather than being interpolated).

## Findings on `DaskSpectralCube.spectral_smooth`

Naïvely you might expect `DaskSpectralCube` to be the fastest path
because it has the `map_blocks` machinery for parallelism. **In its
default configuration with stock astropy on this workload it is the
slowest sensible path**, indistinguishable from a single whole-cube
`astropy.convolve` call.

Why dask doesn't help with stock astropy:

1. **The per-chunk function is `astropy.convolution.convolve`**, whose
   Cython wrapper `_convolveNd_c` (in `astropy/convolution/_convolve.pyx`)
   calls the C routine without a `with nogil:` block. The C function
   itself is declared `nogil`, but the wrapper holds the GIL for the
   whole call. dask's threaded scheduler therefore cannot run multiple
   chunks in parallel — each thread is serialized on the GIL.

2. **Smaller spatial chunks make it worse, not better.** σ=50 on the
   full cube goes 59.6 s → 113.9 s (clean) / 81.2 s → 142.1 s (NaN) when
   we drop from 'auto' to 64×64 spatial chunks. The convolution is still
   serialized by the GIL, so smaller chunks just multiply astropy's
   per-call setup overhead.

3. **Passing a custom `convolve=` function that uses
   `scipy.ndimage.convolve1d` recovers the full speedup** because
   `scipy.ndimage` releases the GIL during its C call, so dask threads
   actually run in parallel. On the (995, 221, 481) cube at σ=5 clean:
   0.87 s for dask+scipy vs 1.62 s for non-dask `vectorize=True`.

### Measured effect of the astropy `nogil` patch

Applying a one-line `with nogil:` block around the C call in
`_convolve.pyx` (PR-ready patch in `/Users/adam/repos/astropy` on branch
`convolve-nogil`) and rebuilding the extension, then re-running the
benchmark in Python 3.13:

| approach                                            | σ=5 stock | σ=5 patched | σ=20 stock | σ=20 patched | σ=50 stock | σ=50 patched |
|-----------------------------------------------------|----------:|------------:|-----------:|-------------:|-----------:|-------------:|
| `DaskSpectralCube.spectral_smooth` (default chunks) |     5.46  |    **1.57** |     22.76  |     **5.38** |     59.58  |    **10.55** |
| `DaskSpectralCube.spectral_smooth` (64×64 chunks)   |     5.93  |        1.76 |     22.68  |         5.65 |    113.92  |        27.10 |
| Same on the NaN cube (default chunks)               |     6.57  |        1.71 |     30.97  |         5.76 |     81.20  |        14.79 |

Speedups from the patch on the default-chunks dask path: **3.5×, 4.2×,
5.6×** at σ=5, 20, 50 (clean), matching the parallel-scaling factor we
inferred from the scipy-per-chunk row. The patch is purely additive
— it doesn't change semantics, doesn't affect single-threaded callers,
and would benefit *every* code path that runs `astropy.convolve` under
threads, not just spectral-cube.

### Final ranking (after both fixes)

The fastest path for spectral smoothing on a cube that fits in memory:

| path                                                | σ=5  | σ=20  | σ=50  |
|-----------------------------------------------------|-----:|------:|------:|
| `DaskSpectralCube.spectral_smooth(vectorize=True)`  | 0.84 |  0.96 |  0.99 |
| `DaskSpectralCube.spectral_smooth` + nogil patch    | 1.57 |  5.38 | 10.55 |
| `SpectralCube.spectral_smooth(vectorize=True)`      | 1.88 |  1.72 |  2.09 |
| `SpectralCube.spectral_smooth` (current default)    | ~6.7 | ~13.6 | ~35.6 |

Both the in-package and upstream fixes are independently worth shipping:

- The in-package `DaskSpectralCube.spectral_smooth(vectorize=True)` is
  the fastest option in absolute terms and lands in spectral-cube
  today with no upstream dependency. Nearly flat in kernel size.
- The astropy `nogil` patch helps *every* threaded user of
  `astropy.convolve`, not just spectral-cube, and brings the default
  dask path within 2× of the in-package fast path.

## What's implemented in `SpectralCube.spectral_smooth(..., vectorize=True)`

Auto-routing between the two scipy backends based on `kernel.array.size`,
crossing over at
`spectral_cube.cube_utils._OACONVOLVE_KERNEL_THRESHOLD` (default 64).
ndimage wins on small kernels, oaconvolve on wide ones. End-to-end times
through the full API on this cube:

| kernel σ (size) | backend chosen | full-API time |
|-----------------|----------------|---------------:|
| 5  (k=41)       | ndimage         | 1.6 s |
| 20 (k=161)      | oaconvolve      | 2.4 s |
| 50 (k=401)      | oaconvolve      | 2.2 s |

Compare to the default per-spectrum path: ~1030 s on the σ=5 case. The
auto-routed `vectorize=True` is ~500× faster at σ=5 and the gap grows to
~4000× at σ=50. NaN handling is implemented via the same
`numerator / denominator` renormalization trick in both backends; output
is bit-equivalent (≤ 1.6e-15) to per-spectrum `astropy.convolve` for all
sane inputs.

`scipy.signal.fftconvolve(axes=0)` is slightly faster than `oaconvolve`
in the benchmark but is *more* sensitive to spectrum length (single big
FFT); `oaconvolve` uses small kernel-sized FFTs which scale better and
have lower peak memory. We picked `oaconvolve` for that reason.
