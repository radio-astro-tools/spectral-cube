from __future__ import print_function, absolute_import, division

import six
import warnings
import tempfile
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
import numpy as np
from radio_beam import Beam, Beams

from .. import SpectralCube, StokesSpectralCube, BooleanArrayMask, LazyMask, VaryingResolutionSpectralCube
from .. import cube_utils
from .. utils import BeamWarning

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

    from casatasks import exportfits

    tmpimagefile = tempfile.mktemp() + '.image'
    tmpfitsfile = tempfile.mktemp() + '.fits'
    ia.newimagefromarray(outfile=tmpimagefile,
                         pixels=np.ones([1] * coordsys.naxes()),
                         csys=coordsys.torecord(), log=False)
    exportfits(tmpimagefile, tmpfitsfile)

    return WCS(tmpfitsfile)


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
        ia = casatools.image()
    except ImportError:
        try:
            from taskinit import iatool
            ia = iatool()
        except ImportError:
            raise ImportError("Could not import CASA (casac) and therefore cannot read CASA .image files")

    # use the ia tool to get the file contents
    try:
        ia.open(filename)
    except AssertionError as ex:
        if 'must be of cReqPath type' in str(ex):
            raise IOError("File {0} not found.  Error was: {1}"
                          .format(filename, str(ex)))
        else:
            raise ex

    # read in the data
    if not skipdata:
        # CASA data are apparently transposed.
        data = ia.getchunk().transpose()

    # CASA stores validity of data as a mask
    if not skipvalid:
        valid = ia.getchunk(getmask=True).transpose()

    # transpose is dealt with within the cube object

    # read in coordinate system object
    casa_cs = ia.coordsys()

    wcs = wcs_casa2astropy(ia, casa_cs)

    unit = ia.brightnessunit()

    beam_ = ia.restoringbeam()
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

    # close the ia tool
    ia.close()

    meta = {'filename': filename,
            'BUNIT': unit}


    if wcs.naxis == 3:
        mask = BooleanArrayMask(np.logical_not(valid), wcs)
        if 'beam' in locals():
            cube = SpectralCube(data, wcs, mask, meta=meta, beam=beam)
        elif 'beams' in locals():
            cube = VaryingResolutionSpectralCube(data, wcs, mask, meta=meta, beams=beams)
        else:
            cube = SpectralCube(data, wcs, mask, meta=meta)
        # we've already loaded the cube into memory because of CASA
        # limitations, so there's no reason to disallow operations
        cube.allow_huge_operations = True

    elif wcs.naxis == 4:
        data, wcs = cube_utils._split_stokes(data, wcs)
        mask = {}
        for component in data:
            data_, wcs_slice = cube_utils._orient(data[component], wcs)
            mask[component] = LazyMask(np.isfinite, data=data[component],
                                       wcs=wcs_slice)

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

    return cube
