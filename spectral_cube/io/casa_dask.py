# Numpy + Dask implementation of CASA image data access

from __future__ import print_function, absolute_import, division

import os
from math import ceil, floor
import uuid
import numpy as np

import dask.array

from .casa_low_level_io import getdminfo

__all__ = ['casa_image_dask_reader']


class MemmapWrapper:
    """
    A wrapper class for dask that opens memmap only when __getitem__ is called,
    which prevents issues with too many open files if opening a memmap for all
    chunks at the same time.
    """

    def __init__(self, filename, **kwargs):
        self._filename = filename
        self._kwargs = kwargs
        self.shape = kwargs['shape'][::-1]
        self.dtype = kwargs['dtype']
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        # We open a file manually and return an in-memory copy of the array
        # otherwise the file doesn't get closed properly.
        with open(self._filename) as f:
            return np.memmap(f, mode='readonly', order='F', **self._kwargs).T[item].copy()


class MaskWrapper:
    """
    A wrapper class for dask that opens binary masks from file and returns
    a chunk from it on-the-fly.
    """

    def __init__(self, filename, offset, count, shape):
        self._filename = filename
        self._offset = offset
        self._count = count
        self.shape = shape[::-1]
        self.dtype = np.bool_
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        # The offset is in the final bit array - but fromfile needs to operate
        # by reading in uint8, so we need to make sure we align what we read
        # in to the bytes.
        start = floor(self._offset / 8)
        end = ceil((self._offset + self._count) / 8)
        array_uint8 = np.fromfile(self._filename, dtype=np.uint8,
                                  offset=start, count=end - start)
        array_bits = np.unpackbits(array_uint8, bitorder='little')
        chunk = array_bits[self._offset - start * 8:self._offset + self._count - start * 8]
        return chunk.reshape(self.shape[::-1], order='F').T[item].astype(np.bool_)


def from_array_fast(arrays, asarray=False, lock=False):
    """
    This is a more efficient alternative to doing::

        [dask.array.from_array(array) for array in arrays]

    that avoids a lot of the overhead in from_array by using the Array
    initializer directly.
    """
    slices = tuple(slice(0, size) for size in arrays[0].shape)
    chunk = tuple((size,) for size in arrays[0].shape)
    meta = np.zeros((0,), dtype=arrays[0].dtype)
    dask_arrays = []
    for array in arrays:
        name1 = str(uuid.uuid4())
        name2 = str(uuid.uuid4())
        dsk = {(name1,) + (0,) * array.ndim: (dask.array.core.getter, name2,
                                              slices, asarray, lock),
               name2: array}
        dask_arrays.append(dask.array.Array(dsk, name1, chunk, meta=meta, dtype=array.dtype))
    return dask_arrays


def casa_image_dask_reader(imagename, memmap=True, mask=False):
    """
    Read a CASA image (a folder containing a ``table.f0_TSM0`` file) into a
    numpy array.
    """

    # the data is stored in the following binary file
    # each of the chunks is stored on disk in fortran-order
    if mask:
        if mask is True:
            mask = 'mask0'
        imagename = os.path.join(str(imagename), mask)

    if not os.path.exists(imagename):
        raise FileNotFoundError(imagename)

    # the data is stored in the following binary file
    # each of the chunks is stored on disk in fortran-order
    img_fn = os.path.join(str(imagename), 'table.f0_TSM0')

    # load the metadata from the image table. Note that this uses our own
    # implementation of getdminfo, which is equivalent to
    # from casatools import table
    # tb = table()
    # tb.open(str(imagename))
    # dminfo = tb.getdminfo()
    # tb.close()
    dminfo = getdminfo(str(imagename))

    # Determine whether file is big endian
    big_endian = dminfo['*1']['BIGENDIAN']

    # chunkshape defines how the chunks (array subsets) are written to disk
    chunkshape = tuple(dminfo['*1']['SPEC']['DEFAULTTILESHAPE'])
    chunksize = np.product(chunkshape)

    # the total shape defines the final output array shape
    totalshape = dminfo['*1']['SPEC']['HYPERCUBES']['*1']['CubeShape']

    # we expect that the total size of the array will be determined by finding
    # the number of chunks along each dimension rounded up
    totalsize = np.product(np.ceil(totalshape / chunkshape)) * chunksize

    # the file size helps us figure out what the dtype of the array is
    filesize = os.stat(img_fn).st_size

    # the ratio between these tells you how many chunks must be combined
    # to create a final stack
    stacks = np.ceil(totalshape / chunkshape).astype(int)
    nchunks = int(np.product(stacks))

    # check that the file size is as expected and determine the data dtype
    if mask:
        expected = nchunks * ceil(chunksize / 8)
        if filesize != expected:
            raise ValueError("Unexpected file size for mask, found {0} but "
                             "expected {1}".format(filesize, expected))
    else:
        if filesize == totalsize * 4:
            if big_endian:
                dtype = '>f4'
            else:
                dtype = '<f4'
            itemsize = 4
        elif filesize == totalsize * 8:
            if big_endian:
                dtype = '>f8'
            else:
                dtype = '<f8'
            itemsize = 8
        else:
            raise ValueError("Unexpected file size for data, found {0} but "
                             "expected {1} or {2}".format(filesize, totalsize * 4, totalsize * 8))

    if memmap:
        if mask:
            chunks = [MaskWrapper(img_fn, offset=ii*ceil(chunksize / 8) * 8, count=chunksize,
                                  shape=chunkshape)
                      for ii in range(nchunks)]
        else:
            chunks = [MemmapWrapper(img_fn, dtype=dtype, offset=ii*chunksize*itemsize,
                                    shape=chunkshape)
                      for ii in range(nchunks)]
    else:
        if mask:
            full_array = np.fromfile(img_fn, dtype='uint8')
            full_array = np.unpackbits(full_array, bitorder='little').astype(np.bool_)
            ceil_chunksize = int(ceil(chunksize / 8)) * 8
            chunks = [full_array[ii*ceil_chunksize:(ii+1)*ceil_chunksize][:chunksize].reshape(chunkshape, order='F').T
                      for ii in range(nchunks)]
        else:
            full_array = np.fromfile(img_fn, dtype=dtype)
            chunks = [full_array[ii*chunksize:(ii+1)*chunksize].reshape(chunkshape, order='F').T
                      for ii in range(nchunks)]

    # convert all chunks to dask arrays - and set name and meta appropriately
    # to prevent dask trying to access the data to determine these
    # automatically.
    chunks = from_array_fast(chunks)

    # make a nested list of all chunks then use block() to construct the final
    # dask array.
    def make_nested_list(chunks, stacks):
        chunks = [chunks[i*stacks[0]:(i+1)*stacks[0]] for i in range(len(chunks) // stacks[0])]
        if len(stacks) > 1:
            return make_nested_list(chunks, stacks[1:])
        else:
            return chunks[0]

    chunks = make_nested_list(chunks, stacks)

    dask_array = dask.array.block(chunks)

    # Since the chunks may not divide the array exactly, all the chunks put
    # together may be larger than the array, so we need to get rid of any
    # extraneous padding.
    final_slice = tuple([slice(dim) for dim in totalshape[::-1]])

    return dask_array[final_slice]
