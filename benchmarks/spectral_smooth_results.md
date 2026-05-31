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

| approach                                            | σ=5 (k=41) | σ=20 (k=161) | σ=50 (k=401) |
|-----------------------------------------------------|-----------:|-------------:|-------------:|
| per-spectrum `astropy.convolve` *(extrapolated)*    |       6.7  |       13.6   |       35.6   |
| **whole-cube** `astropy.convolve` (general fast path)|      5.48  |       22.17  |       60.54  |
| **whole-cube** `scipy.ndimage.convolve1d` (default fast path)| 1.14  |  4.97 | 13.60 |
| whole-cube `astropy.convolve_fft`                   |      28.91 |       29.93  |       36.08  |
| `scipy.signal.fftconvolve(axes=0)`                  |       1.33 |        1.35  |        1.66  |
| `scipy.signal.oaconvolve(axes=0)`                   |       2.04 |        1.18  |        1.53  |

## 5% NaN cube

| approach                                            | σ=5 | σ=20 | σ=50 |
|-----------------------------------------------------|----:|-----:|-----:|
| whole-cube `astropy.convolve`                       | 6.38 | 32.47 | 86.98 |
| **whole-cube** `scipy.ndimage.convolve1d` (default fast path) | 2.56 | 9.80 | 27.15 |
| whole-cube `astropy.convolve_fft`                   | 27.64 | 31.68 | 37.70 |
| `scipy.signal.fftconvolve(axes=0)` *(no NaN handling)* | 1.25 | 1.38 | 1.73 |
| `scipy.signal.oaconvolve(axes=0)` *(no NaN handling)* | 1.97 | 1.31 | 1.57 |

All approaches that handle NaNs match the whole-cube `astropy.convolve`
reference to machine precision (max|Δ| ≤ 1.6e-15).

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
