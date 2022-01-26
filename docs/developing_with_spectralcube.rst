.. _doc_developersnotes:

Notes for development using spectral-cube
=========================================
.. currentmodule:: spectral_cube

spectral-cube is flexible and can used within other packages for
development beyond the core package's capabilities. Two significant strengths
are the use of memory-mapping and  the integration with `dask <https://dask.org/>`_
(:ref:`doc_dask`) to efficiently larger than memory data.

This page provides suggestions for developing using spectral-cube in other
packages.

The following two sections give information on standard usage of  :class:`SpectralCube`.
The third discusses usage with dask integration.

Handling large data cubes
-------------------------

spectral-cube is specifically designed for handling larger-than-memory data
and minimizes creating copies of the data. :class:`SpectralCube` uses memory-mapping
and provides options for executing operations with only subsets of the data
(for example, the `how` keyword in `~SpectralCube.moment`).

Masking operations can be performed "lazily", where the computation is completed
only when a view of the underlying boolean mask array is returned (:ref:`doc_masking`).

Further strategies for handling large data is given in :ref:`_doc_handling_large_datasets`.


Parallelizing operations
------------------------

Several operations implemented in :class:`SpectralCube` can be parallelized
using the `joblib <https://joblib.readthedocs.io/en/latest/>`_ package. Builtin methods
in :class:`SpectralCube` with the `parallel` keyword will enable using joblib.

New methods can take advantage of these features by using creating custom functions
to pass to :meth:`SpectralCube.apply_function_parallel_spatial` and
:meth:`SpectralCube.apply_function_parallel_spectral`. These methods expect
functions with a data and mask array input, with optional `**kwargs` that can be
passed, and expect an output array of the same shape as the input.


Unifying large-data handling and parallelization with dask
----------------------------------------------------------

spectral-cube's dask integration unifies many of the above features and further
options leveraging the dask ecosystem. The :ref:`doc_dask` page provides an overview
of general usage and recommended practices, including:

    * Using different dask schedulers (synchronous, threads, and distributed)
    * Triggering dask executions and saving intermediate results to disk
    * Efficiently rechunking large data for parallel operations
    * Loading cubes in CASA image format

For an interactive demonstration of these features, see
the `Guide to Dask Optimization <https://github.com/radio-astro-tools/tutorials/pull/21>`_.
.. TODO: UPDATE THE LINK TO THE TUTORIAL once merged

For further development, we highlight the ability to apply custom functions using dask.
A :class:`DaskSpectralCube` loads the data as a `dask Array <https://docs.dask.org/en/stable/array.html>`_.
Similar to the non-dask :class:`SpectralCube`, custom function can be used with
:meth:`DaskSpectralCube.apply_function_parallel_spectral` and
:meth:`DaskSpectralCube.apply_function_parallel_spatial`. Effectively these are
wrappers on `dask.array.map_blocks <https://docs.dask.org/en/stable/generated/dask.array.map_blocks.html#dask.array.map_blocks>`_
and accept common kwargs.

.. note::
    The dask array can be accessed with `DaskSpectralCube._data` but we discourage
    this as the builtin functions include checks, such as applying the mask to the
    data.

    If you have a use case needing on of dask array's other `operation tools <https://docs.dask.org/en/stable/array-best-practices.html#build-your-own-operations>`_
    please raise an `issue on GitHub <https://github.com/radio-astro-tools/spectral-cube/issues>`_
    so we can add this support!

The :ref:`doc_dask` page gives a basic example of using a custom function. A more
advanced example is shown in the `parallel fitting with dask tutorial <https://github.com/radio-astro-tools/tutorials/pull/12>`_.
This tutorial demonstrates fitting a spectral model to every spectrum in a cube, applied
in parallel over chunks of the data. This fitting example is a guide for using
:meth:`DaskSpectralCube.apply_function_parallel_spectral` with:

    * A change in array shape and dimensions in the output (`drop_axis` and `chunks` in `dask.array.map_blocks <https://docs.dask.org/en/stable/generated/dask.array.map_blocks.html#dask.array.map_blocks>`_)
    * Using dask's `block_info` dictionary in a custom function to track the location of a chunk in the cube

.. TODO: UPDATE THE LINK TO THE TUTORIAL once merged

