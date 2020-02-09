import numpy as np
import shutil

def casa_image_reader(imagename):
    from casatools import table
    tb = table()

    tb.open(imagename)
    dminfo = tb.getdminfo()
    tb.close()

    chunksize = dminfo['*1']['SPEC']['DEFAULTTILESHAPE']
    totalsize = dminfo['*1']['SPEC']['HYPERCUBES']['*1']['CubeShape']
    stacks = totalsize / chunksize
    nchunks = np.product(totalsize) // np.product(chunksize)
    fh = open(f'{imagename}/table.f0_TSM0', 'rb')
    chunks = [np.fromfile(fh, dtype='float32',
                          count=np.product(chunksize)).reshape(chunksize,
                                                               order='F')
              for ii in range(nchunks)]

    rslt = chunks
    rstacks = list(stacks)
    jj = 0
    while len(rstacks) > 0:
        rstacks.pop()
        kk = len(stacks) - jj - 1
        remaining_dims = rstacks
        if len(remaining_dims) == 0:
            assert kk == 0
            rslt = np.concatenate(rslt, 0)
        else:
            cut = np.product(remaining_dims)
            assert cut % 1 == 0
            cut = int(cut)
            rslt = [np.concatenate(rslt[ii::cut], kk) for ii in range(cut)]
        jj += 1

    return rslt

def og_casa_image_reader(imagename):
    from casatools import image
    ia = image()

    ia.open(imagename)
    ch = ia.getchunk()
    ia.close()
    return ch


def test_casa_image_reader():
    from casatools import image, table
    ia = image()
    tb = table()

    shape = [129,130,128]
    size = np.product(shape)
    im = np.arange(size).reshape(shape)

    shutil.rmtree('test.image')
    ia.fromarray(outfile='test.image', pixels=im, overwrite=True)
    ia.close()

    tb.open('test.image')
    dminfo = tb.getdminfo()
    tb.close()

    chunksize = dminfo['*1']['SPEC']['DEFAULTTILESHAPE']
    totalsize = dminfo['*1']['SPEC']['HYPERCUBES']['*1']['CubeShape']
    stacks = totalsize / chunksize
    nchunks = np.product(totalsize) // np.product(chunksize)
    fh = open('test.image/table.f0_TSM0', 'rb')
    chunks = [np.fromfile(fh, dtype='float32',
                          count=np.product(chunksize)).reshape(chunksize,
                                                               order='F')
              for ii in range(nchunks)]

    # ch1 = [np.concatenate(chunks[ii::20], 0) for ii in range(20)]
    # ch2 = [np.concatenate(ch1[ii::4], 1) for ii in range(4)]
    # ch3 = np.concatenate(ch2, 2)
    # print("forward-order: ",(ch3==im).sum())

    ch1 = [np.concatenate(chunks[ii::15], 2) for ii in range(15)]
    ch2 = [np.concatenate(ch1[ii::3], 1) for ii in range(3)]
    ch3 = np.concatenate(ch2, 0)
    print(ch3.shape, im.shape)
    print("backward-order: ",(ch3==im).sum(),(ch3==im).sum() / im.size)

    #chunkarr = np.array(chunks)
    #print(chunkarr.shape, chunkarr.size)
    #print(shape, size)
    #
    ia.open('test.image')
    getchunk = ia.getchunk()
    ia.close()

    assert np.all(getchunk == im)
    #
    #import itertools
    #for ax1,ax2 in itertools.product((0,1,2,3,),(0,1,2,3,)):
    #    ok = (chunkarr.T.swapaxes(ax1,ax2).reshape(shape, order='A') == im).sum()
    #    print(ax1,ax2,ok)

    assert np.all(casa_image_reader('test.image') == im)
