from __future__ import print_function, absolute_import, division

import warnings
from astropy import units as u
from astropy.io import registry as io_registry
from radio_beam import Beam, Beams

from .. import SpectralCube, StokesSpectralCube, BooleanArrayMask, VaryingResolutionSpectralCube
from ..spectral_cube import BaseSpectralCube
from .. import cube_utils
from .. utils import BeamWarning
from .. import wcs_utils

from .casa_low_level_io import getdesc
from .casa_wcs import wcs_casa2astropy
from .casa_dask import casa_image_dask_reader

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

    if 'units' in desc['_keywords_']:
        unit = desc['_keywords_']['units']
    else:
        unit = ''

    imageinfo = desc['_keywords_']['imageinfo']

    if 'perplanebeams' in imageinfo:
        beam_ = {'beams': imageinfo['perplanebeams']}
        beam_['nStokes'] = beam_['beams'].pop('nStokes')
        beam_['nChannels'] = beam_['beams'].pop('nChannels')
        beam_['beams'] = {key: {'*0': value} for key, value in list(beam_['beams'].items())}
    elif 'restoringbeam' in imageinfo:
        beam_ = imageinfo['restoringbeam']
    else:
        beam_ = {}

    wcs = wcs_casa2astropy(casa_cs)

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
        if mask['I'] is not None:
            assert cube.I.mask.shape == cube.shape
            assert wcs_utils.check_equality(cube.I.mask._wcs, cube.wcs)
    else:
        raise ValueError("CASA image has {0} dimensions, and therefore "
                         "is not readable by spectral-cube.".format(wcs.naxis))

    from .core import normalize_cube_stokes
    return normalize_cube_stokes(cube, target_cls=target_cls)


io_registry.register_reader('casa', BaseSpectralCube, load_casa_image)
io_registry.register_reader('casa_image', BaseSpectralCube, load_casa_image)
io_registry.register_identifier('casa', BaseSpectralCube, is_casa_image)

io_registry.register_reader('casa', StokesSpectralCube, load_casa_image)
io_registry.register_reader('casa_image', StokesSpectralCube, load_casa_image)
io_registry.register_identifier('casa', StokesSpectralCube, is_casa_image)
