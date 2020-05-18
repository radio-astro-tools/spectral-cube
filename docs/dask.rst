Integration with dask
=====================

Getting started
---------------

When loading a cube with the :class:`~spectral_cube.SpectralCube` class, it is possible to optionally
specify the ``use_dask`` keyword argument to control whether or not to use new experimental classes
(:class:`~spectral_cube.DaskSpectralCube` and :class:`~spectral_cube.DaskVaryingResolutionSpectralCube`)
that use `dask <https://dask.org/>`_ for representing cubes and carrying out computations efficiently. The default is
``use_dask=True`` when reading in CASA spectral cubes, but not when loading cubes from other formats.

To read in a FITS cube using the dask-enabled classes, you can do::

    >>> from astropy.utils import data
    >>> from spectral_cube import SpectralCube
    >>> fn = data.get_pkg_data_filename('tests/data/example_cube.fits', 'spectral_cube')
    >>> cube = SpectralCube.read(fn, use_dask=True)
    >>> cube
    DaskSpectralCube with shape=(7, 4, 3) and unit=Jy / beam:
     n_x:      3  type_x: RA---ARC  unit_x: deg    range:    52.231466 deg:   52.231544 deg
     n_y:      4  type_y: DEC--ARC  unit_y: deg    range:    31.243639 deg:   31.243739 deg
     n_s:      7  type_s: VRAD      unit_s: m / s  range:    14322.821 m / s:   14944.909 m / s

Most of the properties and methods that normally work with :class:`~spectral_cube.SpectralCube`
should continue to work with :class:`~spectral_cube.DaskSpectralCube`.

Schedulers and parallel computations
------------------------------------

By default, we use the ``'synchronous'`` `dask scheduler <https://docs.dask.org/en/latest/scheduler-overview.html>`_
which means that calculations are run in a single process/thread. However, you can control this using the
:meth:`~spectral_cube.DaskSpectralCube.use_dask_scheduler` method:

    >>> cube.use_dask_scheduler('threads')  # doctest: +IGNORE_OUTPUT

Any calculation after this will then use the multi-threaded scheduler. It is also possible to use this
as a context manager, to temporarily change the scheduler::

    >>> with cube.use_dask_scheduler('threads'):  # doctest: +IGNORE_OUTPUT
    ...     cube.max()

You can optionally specify the number of threads/processes to use with ``num_workers``::

    >>> with cube.use_dask_scheduler('threads', num_workers=4):  # doctest: +IGNORE_OUTPUT
    ...     cube.max()

If you don't specify the number of threads, this could end up being quite large, and cause you to
run out of memory for certain operations.

Note that running operations in parallel may sometimes be less efficient than running them in
serial depending on how your data is stored, so don't assume that it will always be faster.

If you want to see a progress bar when carrying out calculations, you can make use of the
`dask.diagnostics <https://docs.dask.org/en/latest/diagnostics-local.html>`_ sub-package - run
the following at the start of your script/session, and all subsequent calculations will display
a progress bar:

    >>> from dask.diagnostics import ProgressBar
    >>> pbar = ProgressBar()
    >>> pbar.register()
    >>> cube.max()  # doctest: +IGNORE_OUTPUT
    [########################################] | 100% Completed |  0.1s
    <Quantity 0.01936739 Jy / beam>

Performance benefits of using dask spectral cube classes

Saving intermediate results to disk
-----------------------------------

When calling methods such as for example :meth:`~spectral_cube.DaskSpectralCube.convolve_to` or any other
methods that return a cube, the result is not immediately calculated - instead, the result is only computed
when data is accessed directly (for example via `~spectral_cube.DaskSpectralCube.filled_data`), or when
writing the cube to disk, for example as a FITS file. However, when doing several operations in a row, such
as spectrally smoothing the cube, then spatially smoothing it, it can be more efficient to store intermediate
results to disk. All methods that return a cube can therefore take the ``save_to_tmp_dir`` option (defaulting
to `False`) which can be set to `True` to compute the result of the operation immediately, save it to a
temporary directory, and re-read it immediately from disk (for users interested in how the data is stored,
it is stored as a `zarr <https://zarr.readthedocs.io/en/stable/>`_ dataset)::

    >>> cube_new = cube.sigma_clip_spectrally(3, save_to_tmp_dir=True)  # doctest: +IGNORE_OUTPUT
    [########################################] | 100% Completed |  0.1s
    >>> cube_new
    DaskSpectralCube with shape=(7, 4, 3) and unit=Jy / beam:
     n_x:      3  type_x: RA---ARC  unit_x: deg    range:    52.231466 deg:   52.231544 deg
     n_y:      4  type_y: DEC--ARC  unit_y: deg    range:    31.243639 deg:   31.243739 deg
     n_s:      7  type_s: VRAD      unit_s: m / s  range:    14322.821 m / s:   14944.909 m / s

Note that this requires the `zarr`_ and `fsspec <https://pypi.org/project/fsspec/>`_ packages to be
installed.

This can also be beneficial if you are using multiprocessing or multithreading to carry out calculations,
because zarr works nicely with disk access from different threads and processes.

Performance benefits of dask classes
------------------------------------

The :class:`~spectral_cube.DaskSpectralCube` class provides in general better
performance than the regular :class:`~spectral_cube.SpectralCube` class. As an
example, we take a look at a spectral cube in FITS format for which we want to
determine the continuum using sigma clipping. When doing this in serial mode,
we already see improvements in performance - first we show the regular spectral
cube capabilities without dask::

    >>> from spectral_cube import SpectralCube
    >>> cube_plain = SpectralCube.read('large_spectral_cube.fits')  # doctest: +SKIP
    >>> %time cube_plain.sigma_clip_spectrally(1)  # doctest: +SKIP
    ...
    CPU times: user 5min 58s, sys: 38 s, total: 6min 36s
    Wall time: 6min 37s

and using the :class:`~spectral_cube.DaskSpectralCube` class::

    >>> cube_dask = SpectralCube.read('large_spectral_cube.fits', use_dask=True)  # doctest: +SKIP
    >>> %time cube_dask.sigma_clip_spectrally(1, save_to_tmp_dir=True)  # doctest: +SKIP
    ...
    CPU times: user 51.7 s, sys: 1.29 s, total: 52.9 s
    Wall time: 51.5 s

Using the parallel options mentioned above results in even better performance::

    >>> cube_dask.use_dask_scheduler('threads', num_workers=4)  # doctest: +SKIP
    >>> %time cube_dask.sigma_clip_spectrally(1, save_to_tmp_dir=True)  # doctest: +SKIP
    ...
    CPU times: user 1min 9s, sys: 1.44 s, total: 1min 11s
    Wall time: 18.5 s

In this case, the wall time is 3x faster (and 21x faster than the regular
spectral cube class without dask).

Applying custom functions to cubes
----------------------------------

Like the :class:`~spectral_cube.SpectralCube` class, the
:class:`~spectral_cube.DaskSpectralCube` and
:class:`~spectral_cube.DaskVaryingResolutionSpectralCube` classes have methods for applying custom
functions to all the spectra or all the spatial images in a cube:
:meth:`~spectral_cube.DaskSpectralCube.apply_function_parallel_spectral` and
:meth:`~spectral_cube.DaskSpectralCube.apply_function_parallel_spatial`. By default, these methods
take functions that apply to individual spectra or images, but this can be quite slow for large
spectral cubes. If possible, you should consider supplying a function that can accept 3-d cubes
and operate on all spectra or image slices in a vectorized way.

To demonstrate this, we will read in a mid-sized CASA dataset with 623 channels and 768x768 pixels in
the image plane::

    >>> large = SpectralCube.read('large_spectral_cube.image', format='casa_image', use_dask=True)  # doctest: +SKIP
    >>> large  # doctest: +SKIP
    DaskVaryingResolutionSpectralCube with shape=(623, 768, 768) and unit=Jy / beam:
    n_x:    768  type_x: RA---SIN  unit_x: deg    range:   290.899389 deg:  290.932404 deg
    n_y:    768  type_y: DEC--SIN  unit_y: deg    range:    14.501466 deg:   14.533425 deg
    n_s:    623  type_s: FREQ      unit_s: Hz     range: 216201517973.483 Hz:216277445708.200 Hz

As an example, we will apply sigma clipping to all spectra in the cube. Note that there is a method
to do this (:meth:`~spectral_cube.DaskSpectralCube.sigma_clip_spectrally`) but for the purposes of
demonstration, we will set up the function ourselves and apply it with
:meth:`~spectral_cube.DaskSpectralCube.apply_function_parallel_spectral`. We will use the
:func:`~astropy.stats.sigma_clip` function from astropy::

    >>> from astropy.stats import sigma_clip

By default, this function returns masked arrays, but to apply this to our
spectral cube, we need it to return a plain Numpy array with NaNs for the masked
values. In addition, the original function tends to return warnings we want to
silence, so we can do this here too::

    >>> import warnings
    >>> import numpy as np
    >>> def sigma_clip_with_nan(*args, **kwargs):
    ...     with warnings.catch_warnings():
    ...         warnings.simplefilter('ignore')
    ...         return sigma_clip(*args, axis=0, **kwargs).filled(np.nan)

The ``axis=0`` is so that if the function is passed a cube, it will still work properly.

Let's now call :meth:`~spectral_cube.DaskSpectralCube.apply_function_parallel_spectral`, including the
``save_to_tmp_dir`` option mentioned previously to force the calculation and the storage of the result
to disk::

    >>> clipped_cube = large.apply_function_parallel_spectral(sigma_clip_with_nan, sigma=3,
    ...                                                       save_to_tmp_dir=True)  # doctest: +SKIP
    [########################################] | 100% Completed |  1min 42.3s

The ``sigma`` argument is passed to the ``sigma_clip_with_nan`` function. We now call this
again but specifying that the ``sigma_clip_with_nan`` function can also take cubes, using
the ``accepts_chunks=True`` option (note that for this to work properly, the wrapped function
needs to include ``axis=0`` in the call to :func:`~astropy.stats.sigma_clip` as shown above)::

    >>> clipped_cube = large.apply_function_parallel_spectral(sigma_clip_with_nan, sigma=3,
    ...                                                       accepts_chunks=True,
    ...                                                       save_to_tmp_dir=True)  # doctest: +SKIP
    [########################################] | 100% Completed | 56.8s

This leads to an improvement in performance of 1.8x in this case.
