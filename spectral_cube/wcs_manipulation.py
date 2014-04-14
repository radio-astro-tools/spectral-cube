import numpy as np
from astropy.wcs import WCS

wcs_parameters_to_preserve = ['cel_offset','dateavg','dateobs','equinox',
                              'latpole', 'lonpole', 'mjdavg', 'mjdobs', 'name',
                              'obsgeo', 'phi0', 'radesys', 'restfrq',
                              'restwav', 'specsys', 'ssysobs', 'ssyssrc',
                              'theta0', 'velangl', 'velosys', 'zsource']
# not writable:
# 'lat', 'lng', 'lattyp', 'lngtyp',
                            

def drop_axis(wcs, dropax):
    """
    Drop the ax on axis dropax

    Remove an axis from the WCS
    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS with naxis to be chopped to naxis-1
    dropax: int
        The index of the WCS to drop, counting from 0 (i.e., python convention,
        not FITS convention)
    """
    inds = range(wcs.wcs.naxis)
    inds.pop(dropax)
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def add_stokes_axis_to_wcs(wcs, add_before_ind):
    """
    Add a new Stokes axis that is uncorrelated with any other axes

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to add to
    add_before_ind: int
        Index of the WCS to insert the new Stokes axis in front of.
        To add at the end, do add_before_ind = wcs.wcs.naxis
    """

    naxin = wcs.wcs.naxis
    naxout = naxin+1

    inds = range(naxout)
    inds.pop(add_before_ind)
    inds = np.array(inds)

    outwcs = WCS(naxis=naxout)
    for par in wcs_parameters_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs,par))

    pc = np.zeros([naxout,naxout])
    pc[inds[:,np.newaxis],inds[np.newaxis,:]] = wcs.wcs.get_pc()
    pc[add_before_ind,add_before_ind] = 1

    def insert_at_index(val, index, lst):
        """ insert a value at index into a list """
        return list(lst)[:index] + [val] + list(lst)[index:]

    outwcs.wcs.crpix = insert_at_index(1, add_before_ind, wcs.wcs.crpix)
    outwcs.wcs.cdelt = insert_at_index(1, add_before_ind, wcs.wcs.get_cdelt())
    outwcs.wcs.crval = insert_at_index(1, add_before_ind, wcs.wcs.crval)
    outwcs.wcs.cunit = insert_at_index("", add_before_ind, wcs.wcs.cunit)
    outwcs.wcs.ctype = insert_at_index("STOKES", add_before_ind, wcs.wcs.ctype)
    outwcs.wcs.cname = insert_at_index("STOKES", add_before_ind, wcs.wcs.cname)
    outwcs.wcs.pc = pc

    return outwcs


def wcs_swapaxes(wcs, ax0, ax1):
    """
    Swap axes in a WCS

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to have its axes swapped
    ax0: int
    ax1: int
        The indices of the WCS to be swapped, counting from 0 (i.e., python
        convention, not FITS convention)
    """
    inds = range(wcs.wcs.naxis)
    inds[ax0],inds[ax1] = inds[ax1],inds[ax0]
    inds = np.array(inds)

    return reindex_wcs(wcs, inds)


def reindex_wcs(wcs, inds):
    """
    Re-index a WCS given indices.  The number of axes may be reduced.

    Parameters
    ----------
    wcs: astropy.wcs.WCS
        The WCS to be manipulated
    inds: np.array(dtype='int')
        The indices of the array to keep in the output.
        e.g. swapaxes: [0,2,1,3]
        dropaxes: [0,1,3]
    """
    if not isinstance(inds, np.ndarray):
        raise TypeError("Indices must be an ndarray")
    if inds.dtype.kind != 'i':
        raise TypeError('Indices must be integers')

    outwcs = WCS(naxis=len(inds))
    for par in wcs_parameters_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs,par))

    cdelt = wcs.wcs.get_cdelt()
    pc = wcs.wcs.get_pc()

    outwcs.wcs.crpix = wcs.wcs.crpix[inds]
    outwcs.wcs.cdelt = cdelt[inds]
    outwcs.wcs.crval = wcs.wcs.crval[inds]
    outwcs.wcs.cunit = [wcs.wcs.cunit[i] for i in inds]
    outwcs.wcs.ctype = [wcs.wcs.ctype[i] for i in inds]
    outwcs.wcs.cname = [wcs.wcs.cname[i] for i in inds]
    outwcs.wcs.pc = pc[inds[:,None],inds[None,:]]

    return outwcs


def test_wcs_dropping():
    wcs = WCS(naxis=4)
    wcs.wcs.pc = np.zeros([4,4])
    np.fill_diagonal(wcs.wcs.pc, np.arange(1,5))
    pc = wcs.wcs.pc # for later use below

    dropped = drop_axis(wcs,0)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([2,3,4]))
    dropped = drop_axis(wcs,1)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1,3,4]))
    dropped = drop_axis(wcs,2)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1,2,4]))
    dropped = drop_axis(wcs,3)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1,2,3]))

    wcs = WCS(naxis=4)
    wcs.wcs.cd = pc

    dropped = drop_axis(wcs,0)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([2,3,4]))
    dropped = drop_axis(wcs,1)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1,3,4]))
    dropped = drop_axis(wcs,2)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1,2,4]))
    dropped = drop_axis(wcs,3)
    assert np.all(dropped.wcs.get_pc().diagonal() == np.array([1,2,3]))

def test_wcs_swapping():
    wcs = WCS(naxis=4)
    wcs.wcs.pc = np.zeros([4,4])
    np.fill_diagonal(wcs.wcs.pc, np.arange(1,5))
    pc = wcs.wcs.pc # for later use below

    swapped = wcs_swapaxes(wcs,0,1)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([2,1,3,4]))
    swapped = wcs_swapaxes(wcs,0,3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([4,2,3,1]))
    swapped = wcs_swapaxes(wcs,2,3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([1,2,4,3]))

    wcs = WCS(naxis=4)
    wcs.wcs.cd = pc

    swapped = wcs_swapaxes(wcs,0,1)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([2,1,3,4]))
    swapped = wcs_swapaxes(wcs,0,3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([4,2,3,1]))
    swapped = wcs_swapaxes(wcs,2,3)
    assert np.all(swapped.wcs.get_pc().diagonal() == np.array([1,2,4,3]))

def test_add_stokes():
    wcs = WCS(naxis=3)
    
    for ii in range(4):
        outwcs = add_stokes_axis_to_wcs(wcs,ii)
        assert outwcs.wcs.naxis == 4
