# Pure Python + Numpy implementation of CASA's getdminfo() and getdesc()
# functions for reading metadata about .image files.

import os
import struct
from io import BytesIO
from collections import OrderedDict

import numpy as np

__all__ = ['getdminfo', 'getdesc']

TYPES = ['bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float',
         'double', 'complex', 'dcomplex', 'string', 'table', 'arraybool',
         'arraychar', 'arrayuchar', 'arrayshort', 'arrayushort', 'arrayint',
         'arrayuint', 'arrayfloat', 'arraydouble', 'arraycomplex',
         'arraydcomplex', 'arraystr', 'record', 'other']


def with_nbytes_prefix(func):
    def wrapper(f, *args):
        start = f.tell()
        nbytes = int(read_int32(f))
        if nbytes == 0:
            return
        b = BytesIO(f.read(nbytes - 4))
        result = func(b, *args)
        end = f.tell()
        if end - start != nbytes:
            raise IOError('Function {0} read {1} bytes instead of {2}'
                          .format(func, end - start, nbytes))
        return result
    return wrapper


def read_bool(f):
    return f.read(1) == b'\x01'


def read_int32(f):
    return np.int32(struct.unpack('>i', f.read(4))[0])


def read_int64(f):
    return np.int64(struct.unpack('>q', f.read(8))[0])


def read_float32(f):
    return np.float32(struct.unpack('>f', f.read(4))[0])


def read_float64(f):
    return np.float64(struct.unpack('>d', f.read(8))[0])


def read_complex64(f):
    return np.complex64(read_float32(f) + 1j * read_float32(f))


def read_complex128(f):
    return np.complex128(read_float64(f) + 1j * read_float64(f))


def read_string(f):
    value = read_int32(f)
    return f.read(int(value)).replace(b'\x00', b'').decode('ascii')


@with_nbytes_prefix
def read_iposition(f):

    stype, sversion = read_type(f)

    if stype != 'IPosition' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    nelem = read_int32(f)

    return np.array([read_int32(f) for i in range(nelem)])


ARRAY_ITEM_READERS = {
    'float': ('float', read_float32, np.float32),
    'double': ('double', read_float64, np.float64),
    'dcomplex': ('void', read_complex128, np.complex128),
    'string': ('String', read_string, '<U16'),
    'int': ('Int', read_int32, np.int32)
}


@with_nbytes_prefix
def read_array(f, arraytype):

    typerepr, reader, dtype = ARRAY_ITEM_READERS[arraytype]

    stype, sversion = read_type(f)

    if stype != f'Array<{typerepr}>' or sversion != 3:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    ndim = read_int32(f)
    shape = [read_int32(f) for i in range(ndim)]
    size = read_int32(f)

    values = [reader(f) for i in range(size)]

    return np.array(values, dtype=dtype).reshape(shape)


def read_type(f):
    tp = read_string(f)
    version = read_int32(f)
    return tp, version


@with_nbytes_prefix
def read_record(f):

    stype, sversion = read_type(f)

    if stype != 'Record' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    read_record_desc(f)

    # Not sure what the following value is
    read_int32(f)


@with_nbytes_prefix
def read_record_desc(f):

    stype, sversion = read_type(f)

    if stype != 'RecordDesc' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    # Not sure what the following value is
    nrec = read_int32(f)

    records = OrderedDict()

    for i in range(nrec):
        name = read_string(f)
        rectype = TYPES[read_int32(f)]
        records[name] = {'type': rectype}
        # Here we don't actually load in the data for may of the types - hence
        # why we don't do anything with the values we read in.
        if rectype in ('bool', 'int', 'uint', 'float', 'double',
                       'complex', 'dcomplex', 'string'):
            f.read(4)
        elif rectype == 'table':
            f.read(8)
        elif rectype.startswith('array'):
            read_iposition(f)
            f.read(4)
        elif rectype == 'record':
            read_record_desc(f)
            read_int32(f)
        else:
            raise NotImplementedError("Support for type {0} in RecordDesc not implemented".format(rectype))

    return records


@with_nbytes_prefix
def read_table_record(f, image_path):

    stype, sversion = read_type(f)

    if stype != 'TableRecord' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    records = read_record_desc(f)

    unknown = read_int32(f)  # noqa

    for name, values in records.items():
        rectype = values['type']
        if rectype == 'bool':
            records[name] = read_bool(f)
        elif rectype == 'int':
            records[name] = int(read_int32(f))
        elif rectype == 'uint':
            records[name] = int(read_int32(f))
        elif rectype == 'float':
            records[name] = float(read_float32(f))
        elif rectype == 'double':
            records[name] = float(read_float64(f))
        elif rectype == 'complex':
            records[name] = complex(read_complex64(f))
        elif rectype == 'dcomplex':
            records[name] = complex(read_complex128(f))
        elif rectype == 'string':
            records[name] = read_string(f)
        elif rectype == 'table':
            records[name] = 'Table: ' + os.path.abspath(os.path.join(image_path, read_string(f)))
        elif rectype == 'arrayint':
            records[name] = read_array(f, 'int')
        elif rectype == 'arrayfloat':
            records[name] = read_array(f, 'float')
        elif rectype == 'arraydouble':
            records[name] = read_array(f, 'double')
        elif rectype == 'arraycomplex':
            records[name] = read_array(f, 'complex')
        elif rectype == 'arraydcomplex':
            records[name] = read_array(f, 'dcomplex')
        elif rectype == 'arraystr':
            records[name] = read_array(f, 'string')
        elif rectype == 'record':
            records[name] = read_table_record(f, image_path)
        else:
            raise NotImplementedError("Support for type {0} in TableRecord not implemented".format(rectype))

    return dict(records)


@with_nbytes_prefix
def read_table(f, image_path):

    stype, sversion = read_type(f)

    if stype != 'Table' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    nrow = read_int32(f)
    fmt = read_int32(f)  # noqa
    name = read_string(f)  # noqa

    big_endian = fmt == 0  # noqa

    table_desc = read_table_desc(f, nrow, image_path)

    return table_desc


def read_column_desc(f, image_path):

    unknown = read_int32(f)  # noqa

    stype, sversion = read_type(f)

    if not stype.startswith(('ScalarColumnDesc', 'ArrayColumnDesc')) or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    desc = {}
    name = read_string(f)
    desc['comment'] = read_string(f)
    desc['dataManagerType'] = read_string(f).replace('Shape', 'Cell')
    desc['dataManagerGroup'] = read_string(f)
    desc['valueType'] = TYPES[read_int32(f)]
    desc['maxlen'] = read_int32(f)
    ndim = read_int32(f)
    if ndim > 0:
        ipos = read_iposition(f)  # noqa
        desc['ndim'] = ndim
    desc['option'] = read_int32(f)
    desc['keywords'] = read_table_record(f, image_path)
    if desc['valueType'] in ('ushort', 'short'):
        f.read(2)
    if desc['valueType'] in ('uint', 'int', 'float', 'string'):
        f.read(4)
    elif desc['valueType'] in ('double', 'complex'):
        f.read(8)
    elif desc['valueType'] in ('dcomplex'):
        f.read(16)
    return {name: desc}


@with_nbytes_prefix
def read_table_desc(f, nrow, image_path):

    stype, sversion = read_type(f)

    if stype != 'TableDesc' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    unknown1 = read_int32(f)  # noqa
    unknown2 = read_int32(f)  # noqa
    unknown3 = read_int32(f)  # noqa

    desc = {}

    desc['_keywords_'] = read_table_record(f, image_path)
    desc['_define_hypercolumn_'] = {}

    hypercolumn = read_table_record(f, image_path)
    desc['_private_keywords_'] = hypercolumn
    if hypercolumn:
        name = list(hypercolumn)[0].split('_')[1]
        value = list(hypercolumn.values())[0]
        desc['_define_hypercolumn_'][name] = {'HCcoordnames': value['coord'],
                                                'HCdatanames': value['data'],
                                                'HCidnames': value['id'],
                                                'HCndim': value['ndim']}

    ncol = read_int32(f)

    for icol in range(ncol):
        if icol > 0:
            read_int32(f)
        array_column_desc = read_column_desc(f, image_path)
        desc.update(array_column_desc)

    return desc


@with_nbytes_prefix
def read_tiled_st_man(f):

    # The code in this function corresponds to TiledStMan::headerFileGet
    # https://github.com/casacore/casacore/blob/75b358be47039250e03e5042210cbc60beaaf6e4/tables/DataMan/TiledStMan.cc#L1086

    stype, sversion = read_type(f)

    if stype != 'TiledStMan' or sversion != 2:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    st_man = {}
    st_man['SPEC'] = {}

    st_man['BIGENDIAN'] = f.read(1) == b'\x01'  # noqa

    seqnr = read_int32(f)
    if seqnr != 0:
        raise ValueError("Expected seqnr to be 0, got {0}".format(seqnr))
    st_man['SEQNR'] = seqnr
    st_man['SPEC']['SEQNR'] = seqnr

    nrows = read_int32(f)
    if nrows != 1:
        raise ValueError("Expected nrows to be 1, got {0}".format(nrows))

    ncols = read_int32(f)
    if ncols != 1:
        raise ValueError("Expected ncols to be 1, got {0}".format(ncols))

    dtype = read_int32(f)  # noqa

    column_name = read_string(f)
    st_man['COLUMNS'] = np.array([column_name], dtype='<U16')
    st_man['NAME'] = column_name

    max_cache_size = read_int32(f)
    st_man['SPEC']['MAXIMUMCACHESIZE'] = max_cache_size
    st_man['SPEC']['MaxCacheSize'] = max_cache_size

    ndim = read_int32(f)

    nrfile = read_int32(f)  # 1
    if nrfile != 1:
        raise ValueError("Expected nrfile to be 1, got {0}".format(nrfile))

    # The following flag seems to control whether or not the TSM file is
    # opened by CASA, and is probably safe to ignore here.
    flag = bool(f.read(1))

    # The following two values are unknown, but are likely relevant when there
    # are more that one field in the image.

    mode = read_int32(f)
    unknown = read_int32(f)  # 0

    bucket = st_man['SPEC']['HYPERCUBES'] = {}
    bucket = st_man['SPEC']['HYPERCUBES']['*1'] = {}

    if mode == 1:
        total_cube_size = read_int32(f)
    elif mode == 2:
        total_cube_size = read_int64(f)
    else:
        raise ValueError('Unexpected value {0} at position {1}'.format(mode, f.tell() - 8))

    unknown = read_int32(f)  # 1
    unknown = read_int32(f)  # 1

    read_record(f)

    flag = f.read(1)  # noqa

    ndim = read_int32(f)  # noqa

    bucket['CubeShape'] = bucket['CellShape'] = read_iposition(f)
    bucket['TileShape'] = read_iposition(f)
    bucket['ID'] = {}
    bucket['BucketSize'] = int(total_cube_size / np.product(np.ceil(bucket['CubeShape'] / bucket['TileShape'])))

    unknown = read_int32(f)  # noqa
    unknown = read_int32(f)  # noqa

    st_man['TYPE'] = 'TiledCellStMan'

    return st_man


@with_nbytes_prefix
def read_tiled_cell_st_man(f):

    stype, sversion = read_type(f)

    if stype != 'TiledCellStMan' or sversion != 1:
        raise NotImplementedError('Support for {0} version {1} not implemented'.format(stype, sversion))

    default_tile_shape = read_iposition(f)

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


def getdesc(filename):
    """
    Return the same output as CASA's getdesc() function, namely a dictionary
    with metadata about the .image file, parsed from the ``table.dat`` file.
    """

    with open(os.path.join(filename, 'table.dat'), 'rb') as f:

        magic = f.read(4)
        if magic != b'\xbe\xbe\xbe\xbe':
            raise ValueError('Incorrect magic code: {0}'.format(magic))

        return read_table(f, filename)
