from __future__ import print_function, absolute_import, division

import six
import warnings
import tempfile
import shutil
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.wcs.wcsapi.sliced_low_level_wcs import sanitize_slices
from astropy import log
import numpy as np
from radio_beam import Beam, Beams

import dask.array

from .. import SpectralCube, StokesSpectralCube, BooleanArrayMask, LazyMask, VaryingResolutionSpectralCube
from .. import cube_utils
from .. utils import BeamWarning, cached
from .. import wcs_utils

# Read and write from a CASA image. This has a few
# complications. First, by default CASA does not return the
# "python order" and so we either have to transpose the cube on
# read or have dueling conventions. Second, CASA often has
# degenerate stokes axes present in unpredictable places (3rd or
# 4th without a clear expectation). We need to replicate these
# when writing but don't want them in memory. By default, try to
# yield the same array in memory that we would get from astropy.


def is_casa_image(input, **kwargs):
    if isinstance(input, six.string_types):
        if input.endswith('.image'):
            return True
    return False


def wcs_casa2astropy(ia, coordsys):
    """
    Convert a casac.coordsys object into an astropy.wcs.WCS object
    """

    # Rather than try and parse the CASA coordsys ourselves, we delegate
    # to CASA by getting it to write out a FITS file and reading back in
    # using WCS

    tmpimagefile = tempfile.mktemp() + '.image'
    tmpfitsfile = tempfile.mktemp() + '.fits'
    ia.fromarray(outfile=tmpimagefile,
                 pixels=np.ones([1] * coordsys.naxes()),
                 csys=coordsys.torecord(), log=False)
    ia.done()

    ia.open(tmpimagefile)
    ia.tofits(tmpfitsfile, stokeslast=False)
    ia.done()

    return WCS(tmpfitsfile)


class ArraylikeCasaData:

    def __init__(self, filename, ia_kwargs={}):

        try:
            import casatools
            self.iatool = casatools.image
            tb = casatools.table()
        except ImportError:
            try:
                from taskinit import iatool, tbtool
                self.iatool = iatool
                tb = tbtool()
            except ImportError:
                raise ImportError("Could not import CASA (casac) and therefore cannot read CASA .image files")


        self.ia_kwargs = ia_kwargs

        self.filename = filename

        self._cache = {}

        log.debug("Creating ArrayLikeCasa object")

        # try to trick CASA into destroying the ia object
        def getshape():
            ia = self.iatool()
            # use the ia tool to get the file contents
            try:
                ia.open(self.filename, cache=False)
            except AssertionError as ex:
                if 'must be of cReqPath type' in str(ex):
                    raise IOError("File {0} not found.  Error was: {1}"
                                  .format(self.filename, str(ex)))
                else:
                    raise ex

            self.shape = tuple(ia.shape()[::-1])
            self.dtype = np.dtype(ia.pixeltype())

            ia.done()
            ia.close()

        getshape()

        self.ndim = len(self.shape)

        tb.open(self.filename)
        dminfo = tb.getdminfo()
        tb.done()

        # unclear if this is always the correct callspec!!!
        # (transpose requires this be backwards)
        self.chunksize = dminfo['*1']['SPEC']['DEFAULTTILESHAPE'][::-1]


        log.debug("Finished with initialization of ArrayLikeCasa object")



    def __getitem__(self, value):


        log.debug(f"Retrieving slice {value} from {self}")

        value = sanitize_slices(value[::-1], self.ndim)

        # several cases:
        # integer: just use an integer
        # slice starting w/number: start number
        # slice starting w/None: use -1
        blc = [(-1 if slc.start is None else slc.start)
               if hasattr(slc, 'start') else slc
               for slc in value]
        # slice ending w/number >= 1: end number -1 (CASA is end-inclusive)
        # slice ending w/zero: use zero, not -1.
        # slice ending w/negative: use it, but ?????
        # slice ending w/None: use -1
        trc = [((slc.stop-1 if slc.stop >= 1 else slc.stop)
                if slc.stop is not None else -1)
               if hasattr(slc, 'stop') else slc for slc in value]
        inc = [(slc.step or 1) if hasattr(slc, 'step') else 1 for slc in value]


        ia = self.iatool()
        # use the ia tool to get the file contents
        try:
            ia.open(self.filename, cache=False)
        except AssertionError as ex:
            if 'must be of cReqPath type' in str(ex):
                raise IOError("File {0} not found.  Error was: {1}"
                              .format(self.filename, str(ex)))
            else:
                raise ex

        log.debug(f'blc={blc}, trc={trc}, inc={inc}, kwargs={self.ia_kwargs}')
        data = ia.getchunk(blc=blc, trc=trc, inc=inc, **self.ia_kwargs)
        ia.done()
        ia.close()

        log.debug(f"Done retrieving slice {value} from {self}")

        # keep all sliced axes but drop all integer axes
        new_view = [slice(None) if isinstance(slc, slice) else 0
                    for slc in value]

        transposed_data = data[tuple(new_view)].transpose()

        log.debug(f"Done transposing data with view {new_view}")

        return transposed_data


def load_casa_image(filename, skipdata=False,
                    skipvalid=False, skipcs=False, **kwargs):
    """
    Load a cube (into memory?) from a CASA image. By default it will transpose
    the cube into a 'python' order and drop degenerate axes. These options can
    be suppressed. The object holds the coordsys object from the image in
    memory.
    """

    try:
        import casatools
        iatool = casatools.image
    except ImportError:
        try:
            from taskinit import iatool
        except ImportError:
            raise ImportError("Could not import CASA (casac) and therefore cannot read CASA .image files")

    ia = iatool()

    # use the ia tool to get the file contents
    try:
        ia.open(filename, cache=False)
    except AssertionError as ex:
        if 'must be of cReqPath type' in str(ex):
            raise IOError("File {0} not found.  Error was: {1}"
                          .format(filename, str(ex)))
        else:
            raise ex

    # read in the data
    if not skipdata:
        arrdata = ArraylikeCasaData(filename)
        # CASA data are apparently transposed.
        data = dask.array.from_array(arrdata,
                                     chunks=arrdata.chunksize,
                                     name=filename
                                    )

    # CASA stores validity of data as a mask
    if not skipvalid:
        boolarr = ArraylikeCasaData(filename, ia_kwargs={'getmask': True})
        valid = dask.array.from_array(boolarr, chunks=boolarr.chunksize,
                                      name=filename+".mask"
                                     )

    # transpose is dealt with within the cube object

    # read in coordinate system object
    casa_cs = ia.coordsys()

    unit = ia.brightnessunit()

    beam_ = ia.restoringbeam()

    ia.done()
    ia.close()

    wcs = wcs_casa2astropy(ia, casa_cs)

    del casa_cs
    del ia


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
        valid, _ = cube_utils._orient(valid, wcs)

        mask = BooleanArrayMask(valid, wcs_slice)
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
        assert cube.mask.shape == cube.shape

    elif wcs.naxis == 4:
        valid, _ = cube_utils._split_stokes(valid, wcs)
        data, wcs = cube_utils._split_stokes(data, wcs)
        mask = {}
        for component in data:
            data_, wcs_slice = cube_utils._orient(data[component], wcs)
            valid_, _ = cube_utils._orient(valid[component], wcs)
            mask[component] = BooleanArrayMask(valid_, wcs_slice)

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
        assert cube.I.mask.shape == cube.shape
        assert wcs_utils.check_equality(cube.I.mask._wcs, cube.wcs)
    else:
        raise ValueError("CASA image has {0} dimensions, and therefore "
                         "is not readable by spectral-cube.".format(wcs.naxis))


    return cube
