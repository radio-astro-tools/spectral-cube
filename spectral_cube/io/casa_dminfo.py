# Pure Python + Numpy implementation of CASA's getdminfo() function
# for reading metadata about .image files

import os
import sys
import struct

import numpy as np


def read_int(f):
    return np.int32(struct.unpack('>I', f.read(4))[0])


def read_string(f):
    value = read_int(f)
    return f.read(int(value))


def read_vector(f):
    nelem = read_int(f)
    return np.array([read_int(f) for i in range(nelem)])


def with_nbytes_prefix(func):
    def wrapper(f):
        start = f.tell()
        nbytes = read_int(f)
        result = func(f)
        end = f.tell()
        if end - start != nbytes:
            raise IOError('Function {0} read {1} bytes instead of {2}'
                          .format(func, end - start, nbytes))
        return result
    return wrapper


@with_nbytes_prefix
def _read_iposition(f):

    title = read_string(f)
    assert title == b'IPosition'

    # Not sure what the next value is, seems to always be one.
    # Maybe number of dimensions?
    number = read_int(f)
    assert number == 1

    return read_vector(f)


def read_type(f):
    tp = read_string(f)
    version = read_int(f)
    return tp, version


@with_nbytes_prefix
def read_record(f):

    stype, sversion = read_type(f)

    if stype != b'Record' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    read_record_desc(f)

    # Not sure what the following value is
    read_int(f)


@with_nbytes_prefix
def read_record_desc(f):

    stype, sversion = read_type(f)

    if stype != b'RecordDesc' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    # Not sure what the following value is
    read_int(f)


@with_nbytes_prefix
def read_tiled_st_man(f):

    # The code in this function corresponds to TiledStMan::headerFileGet
    # https://github.com/casacore/casacore/blob/75b358be47039250e03e5042210cbc60beaaf6e4/tables/DataMan/TiledStMan.cc#L1086

    stype, sversion = read_type(f)

    if stype != b'TiledStMan' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    st_man = {}
    st_man['SPEC'] = {}

    big_endian = f.read(1)  # noqa

    seqnr = read_int(f)
    if seqnr != 0:
        raise ValueError("Expected seqnr to be 0, got {0}".format(seqnr))
    st_man['SEQNR'] = seqnr
    st_man['SPEC']['SEQNR'] = seqnr

    nrows = read_int(f)
    if nrows != 1:
        raise ValueError("Expected nrows to be 1, got {0}".format(nrows))

    ncols = read_int(f)
    if ncols != 1:
        raise ValueError("Expected ncols to be 1, got {0}".format(ncols))

    dtype = read_int(f)  # noqa

    column_name = read_string(f).decode('ascii')
    st_man['COLUMNS'] = np.array([column_name], dtype='<U16')
    st_man['NAME'] = column_name

    max_cache_size = read_int(f)
    st_man['SPEC']['MAXIMUMCACHESIZE'] = max_cache_size
    st_man['SPEC']['MaxCacheSize'] = max_cache_size

    ndim = read_int(f)

    nrfile = read_int(f)  # 1
    if nrfile != 1:
        raise ValueError("Expected nrfile to be 1, got {0}".format(nrfile))

    # The following flag seems to control whether or not the TSM file is
    # opened by CASA, and is probably safe to ignore here.
    flag = bool(f.read(1))

    # The following two values are unknown, but are likely relevant when there
    # are more that one field in the image.

    unknown = read_int(f)  # 0
    unknown = read_int(f)  # 0

    bucket = st_man['SPEC']['HYPERCUBES'] = {}
    bucket = st_man['SPEC']['HYPERCUBES']['*1'] = {}

    total_cube_size = read_int(f)  # noqa

    unknown = read_int(f)  # 1
    unknown = read_int(f)  # 1

    read_record(f)

    flag = f.read(1)  # noqa

    ndim = read_int(f)  # noqa

    bucket['CubeShape'] = bucket['CellShape'] = _read_iposition(f)
    bucket['TileShape'] = _read_iposition(f)
    bucket['ID'] = {}
    bucket['BucketSize'] = int(total_cube_size / np.product(np.ceil(bucket['CubeShape'] / bucket['TileShape'])))


    unknown = read_int(f)  # noqa
    unknown = read_int(f)  # noqa

    st_man['TYPE'] = 'TiledCellStMan'

    return st_man


@with_nbytes_prefix
def read_tiled_cell_st_man(f):

    stype, sversion = read_type(f)

    if stype != b'TiledCellStMan' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    default_tile_shape = _read_iposition(f)

    st_man = read_tiled_st_man(f)

    st_man['SPEC']['DEFAULTTILESHAPE'] = default_tile_shape

    return {'*1': st_man}


def getdminfo(filename):
    """
    Return the same output as CASA's getdminfo() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.f0`` file.
    """

    with open(os.path.join(filename, 'table.f0'), 'rb') as f:

        magic = f.read(4)
        if magic != b'\xbe\xbe\xbe\xbe':
            raise ValueError('Incorrect magic code: {0}'.format(magic))

        return read_tiled_cell_st_man(f)
