from __future__ import print_function, absolute_import, division

import os
import six
from math import ceil, floor
import uuid
import warnings
import tempfile
import shutil
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.wcs.wcsapi.sliced_low_level_wcs import sanitize_slices
from astropy import log
from astropy.io import registry as io_registry
import numpy as np
from radio_beam import Beam, Beams

import dask.array

from .. import SpectralCube, StokesSpectralCube, BooleanArrayMask, LazyMask, VaryingResolutionSpectralCube
from ..spectral_cube import BaseSpectralCube
from .. import cube_utils
from .. utils import BeamWarning, cached, StokesWarning
from .. import wcs_utils
from .casa_dminfo import getdminfo, getdesc

# Read and write from a CASA image. This has a few
# complications. First, by default CASA does not return the
# "python order" and so we either have to transpose the cube on
# read or have dueling conventions. Second, CASA often has
# degenerate stokes axes present in unpredictable places (3rd or
# 4th without a clear expectation). We need to replicate these
# when writing but don't want them in memory. By default, try to
# yield the same array in memory that we would get from astropy.


def is_casa_image(origin, filepath, fileobj, *args, **kwargs):

    # See note before StringWrapper definition
    from .core import StringWrapper
    if filepath is None and len(args) > 0:
        if isinstance(args[0], StringWrapper):
            filepath = args[0].value
        elif isinstance(args[0], str):
            filepath = args[0]

    return filepath is not None and filepath.lower().endswith('.image')


def wcs_casa2astropy(ndim, coordsys):
    """
    Convert a casac.coordsys object into an astropy.wcs.WCS object
    """

    # Rather than try and parse the CASA coordsys ourselves, we delegate
    # to CASA by getting it to write out a FITS file and reading back in
    # using WCS

    try:
        from casatools import image
    except ImportError:
        try:
            from taskinit import iatool as image
        except ImportError:
            raise ImportError("Could not import CASA (casac) and therefore cannot convert csys to WCS")

    tmpimagefile = tempfile.mktemp() + '.image'
    tmpfitsfile = tempfile.mktemp() + '.fits'
    ia = image()
    ia.fromarray(outfile=tmpimagefile,
                 pixels=np.ones([1] * ndim),
                 csys=coordsys, log=False)
    ia.done()

    ia.open(tmpimagefile)
    ia.tofits(tmpfitsfile, stokeslast=False)
    ia.done()

    return WCS(tmpfitsfile)


def load_casa_image(filename, skipdata=False,
                    skipvalid=False, skipcs=False, target_cls=None, **kwargs):
    """
    Load a cube (into memory?) from a CASA image. By default it will transpose
    the cube into a 'python' order and drop degenerate axes. These options can
    be suppressed. The object holds the coordsys object from the image in
    memory.
    """

    from .core import StringWrapper
    if isinstance(filename, StringWrapper):
        filename = filename.value

    # read in the data
    if not skipdata:
        data = casa_image_dask_reader(filename)

    # CASA stores validity of data as a mask
    if skipvalid:
        valid = None
    else:
        try:
            valid = casa_image_dask_reader(filename, mask=True)
        except FileNotFoundError:
            valid = None

    # transpose is dealt with within the cube object

    # read in coordinate system object

    desc = getdesc(filename)

    casa_cs = desc['_keywords_']['coords']

    unit = desc['_keywords_']['units']

    if 'perplanebeams' in desc['_keywords_']['imageinfo']:
        beam_ = {'beams': desc['_keywords_']['imageinfo']['perplanebeams']}
        beam_['nStokes'] = beam_['beams'].pop('nStokes')
        beam_['nChannels'] = beam_['beams'].pop('nChannels')
        beam_['beams'] = {key: {'*0': value} for key, value in list(beam_['beams'].items())}
    else:
        beam_ = desc['_keywords_']['imageinfo']['restoringbeam']

    wcs = wcs_casa2astropy(data.ndim, casa_cs)

    del casa_cs

    if 'major' in beam_:
        beam = Beam(major=u.Quantity(beam_['major']['value'], unit=beam_['major']['unit']),
                    minor=u.Quantity(beam_['minor']['value'], unit=beam_['minor']['unit']),
                    pa=u.Quantity(beam_['positionangle']['value'], unit=beam_['positionangle']['unit']),
                   )
    elif 'beams' in beam_:
        bdict = beam_['beams']
        if beam_['nStokes'] > 1:
            raise NotImplementedError()
        nbeams = len(bdict)
        assert nbeams == beam_['nChannels']
        stokesidx = '*0'

        majors = [u.Quantity(bdict['*{0}'.format(ii)][stokesidx]['major']['value'],
                             bdict['*{0}'.format(ii)][stokesidx]['major']['unit']) for ii in range(nbeams)]
        minors = [u.Quantity(bdict['*{0}'.format(ii)][stokesidx]['minor']['value'],
                             bdict['*{0}'.format(ii)][stokesidx]['minor']['unit']) for ii in range(nbeams)]
        pas = [u.Quantity(bdict['*{0}'.format(ii)][stokesidx]['positionangle']['value'],
                          bdict['*{0}'.format(ii)][stokesidx]['positionangle']['unit']) for ii in range(nbeams)]

        beams = Beams(major=u.Quantity(majors),
                      minor=u.Quantity(minors),
                      pa=u.Quantity(pas))
    else:
        warnings.warn("No beam information found in CASA image.",
                      BeamWarning)


    # don't need this yet
    # stokes = get_casa_axis(temp_cs, wanttype="Stokes", skipdeg=False,)

    #    if stokes == None:
    #        order = np.arange(self.data.ndim)
    #    else:
    #        order = []
    #        for ax in np.arange(self.data.ndim+1):
    #            if ax == stokes:
    #                continue
    #            order.append(ax)

    #    self.casa_cs = ia.coordsys(order)

        # This should work, but coordsys.reorder() has a bug
        # on the error checking. JIRA filed. Until then the
        # axes will be reversed from the original.

        # if transpose == True:
        #    new_order = np.arange(self.data.ndim)
        #    new_order = new_order[-1*np.arange(self.data.ndim)-1]
        #    print new_order
        #    self.casa_cs.reorder(new_order)


    meta = {'filename': filename,
            'BUNIT': unit}


    if wcs.naxis == 3:
        data, wcs_slice = cube_utils._orient(data, wcs)

        if valid is not None:
            valid, _ = cube_utils._orient(valid, wcs)
            mask = BooleanArrayMask(valid, wcs_slice)
        else:
            mask = None

        if 'beam' in locals():
            cube = SpectralCube(data, wcs_slice, mask, meta=meta, beam=beam)
        elif 'beams' in locals():
            cube = VaryingResolutionSpectralCube(data, wcs_slice, mask, meta=meta, beams=beams)
        else:
            cube = SpectralCube(data, wcs_slice, mask, meta=meta)
        # with #592, this is no longer true
        # we've already loaded the cube into memory because of CASA
        # limitations, so there's no reason to disallow operations
        # cube.allow_huge_operations = True
        if mask is not None:
            assert cube.mask.shape == cube.shape

    elif wcs.naxis == 4:
        if valid is not None:
            valid, _ = cube_utils._split_stokes(valid, wcs)
        data, wcs = cube_utils._split_stokes(data, wcs)
        mask = {}
        for component in data:
            data_, wcs_slice = cube_utils._orient(data[component], wcs)
            if valid is not None:
                valid_, _ = cube_utils._orient(valid[component], wcs)
                mask[component] = BooleanArrayMask(valid_, wcs_slice)
            else:
                mask[component] = None

            if 'beam' in locals():
                data[component] = SpectralCube(data_, wcs_slice, mask[component],
                                               meta=meta, beam=beam)
            elif 'beams' in locals():
                data[component] = VaryingResolutionSpectralCube(data_,
                                                                wcs_slice,
                                                                mask[component],
                                                                meta=meta,
                                                                beams=beams)
            else:
                data[component] = SpectralCube(data_, wcs_slice, mask[component],
                                               meta=meta)

            data[component].allow_huge_operations = True


        cube = StokesSpectralCube(stokes_data=data)
        if mask is not None:
            assert cube.I.mask.shape == cube.shape
            assert wcs_utils.check_equality(cube.I.mask._wcs, cube.wcs)
    else:
        raise ValueError("CASA image has {0} dimensions, and therefore "
                         "is not readable by spectral-cube.".format(wcs.naxis))

    from .core import normalize_cube_stokes
    return normalize_cube_stokes(cube, target_cls=target_cls)


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
        return chunk.astype(np.bool_).reshape(self.shape[::-1], order='F').T[item]


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

    # chunkshape definse how the chunks (array subsets) are written to disk
    chunkshape = tuple(dminfo['*1']['SPEC']['DEFAULTTILESHAPE'])
    chunksize = np.product(chunkshape)

    # the total size defines the final output array size
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
            dtype = np.float32
            itemsize = 4
        elif filesize == totalsize * 8:
            dtype = np.float64
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


io_registry.register_reader('casa', BaseSpectralCube, load_casa_image)
io_registry.register_reader('casa_image', BaseSpectralCube, load_casa_image)
io_registry.register_identifier('casa', BaseSpectralCube, is_casa_image)

io_registry.register_reader('casa', StokesSpectralCube, load_casa_image)
io_registry.register_reader('casa_image', StokesSpectralCube, load_casa_image)
io_registry.register_identifier('casa', StokesSpectralCube, is_casa_image)
