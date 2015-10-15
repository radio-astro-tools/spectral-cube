Examples
========

1. From a cube with many lines, extract each line and create moment maps using
   the brightest line as a mask:

.. code-block:: python

    import numpy as np
    from spectral_cube import SpectralCube
    from astropy import units as u

    # Read the FITS cube
    # And change the units back to Hz
    # (note that python doesn't care about the line breaks here)
    cube = (SpectralCube
            .read('my_multiline_file.fits')
            .with_spectral_unit(u.Hz))

    # Lines to be analyzed (including brightest_line)
    my_line_list = [362.630304, 364.103249, 363.945894, 363.785397, 362.736048] * u.GHz
    my_line_widths = [150.0, 80.0, 80.0, 80.0, 80.0] * u.km/u.s
    my_line_names = ['HNC43','H2COJ54K4','H2COJ54K2','HC3N4039','H2COJ54K0']
    # These are:
    # H2CO 5(4)-4(4) at 364.103249 GHz
    # H2CO 5(24)-4(23) at 363.945894 GHz
    # HC3N 40-39 at 363.785397 GHz
    # H2CO 5(05)-4(04) at 362.736048 GHz (actually a blend with HNC 4-3...)

    brightest_line = 362.630304*u.GHz # HNC 4-3

    # What is the maximum width spanned by the galaxy (in velocity)
    width = 150*u.km/u.s

    # Velocity center
    vz = 258*u.km/u.s

    # Use the brightest line to identify the appropriate peak velocities, but ONLY
    # from a slab including +/- width:
    brightest_cube = (cube
                      .with_spectral_unit(u.km/u.s, rest_value=brightest_line,
                                          velocity_convention='optical')
                      .spectral_slab(vz-width, vz+width))

    # velocity of the brightest pixel
    peak_velocity = brightest_cube.spectral_axis[brightest_cube.argmax(axis=0)]

    # make a spatial mask excluding pixels with no signal
    peak_amplitude = brightest_cube.max(axis=0)

    # Create a noise map from a line-free region.
    # found this range from inspection of a spectrum:
    # s = cube.max(axis=(1,2))
    # s.quicklook()
    noisemap = cube.spectral_slab(362.603*u.GHz, 363.283*u.GHz).std(axis=0)
    spatial_mask = peak_amplitude > 3*noisemap

    # Now loop over EACH line, extracting moments etc. from the appropriate region:
    # we'll also apply a transition-dependent width (my_line_widths) here because
    # these fainter lines do not have peaks as far out as the bright line.

    for line_name,line_freq,line_width in zip(my_line_names,my_line_list,my_line_widths):

        subcube = cube.with_spectral_unit(u.km/u.s,
                                          rest_value=line_freq,
                                          velocity_convention='optical'
                                         ).spectral_slab(peak_velocity.min()-line_width,
                                                         peak_velocity.max()+line_width)

        # this part makes a cube of velocities for masking work
        temp = subcube.spectral_axis
        velocities = np.tile(temp[:,None,None], subcube.shape[1:])
        # `velocities` has the same shape as `subcube`

        # now we use the velocities from the brightest line to create a mask region
        # in the same velocity range but with different rest frequencies (different
        # lines)
        mask = np.abs(peak_velocity - velocities) < line_width

        # Mask on a pixel-by-pixel basis with a 1-sigma cut
        signal_mask = subcube > noisemap

        # the mask is a cube, the spatial mask is a 2d array, but in this case
        # numpy knows how to combine them properly
        # (signal_mask is a different type, so it can't be combined with the others
        # yet - https://github.com/radio-astro-tools/spectral-cube/issues/231)
        msubcube = subcube.with_mask(mask & spatial_mask).with_mask(signal_mask)

        # Then make & save the moment maps
        for moment in (0,1,2):
            mom = msubcube.moment(order=moment, axis=0)
            mom.hdu.writeto("moment{0}/{1}_{2}_moment{0}.fits".format(moment,target,line_name), clobber=True)



2. Use aplpy (in a slightly unsupported way) to make an RGB velocity movie

.. code-block:: python

   import aplpy

   cube = SpectralCube.read('file.fits')
   prefix = 'HC3N'
   
   # chop out the NaN borders
   cmin = cube.minimal_subcube()
   
   # Create the WCS template
   F = aplpy.FITSFigure(cmin[0].hdu)

   # decide on the velocity range
   v1 = 30*u.km/u.s
   v2 = 60*u.km/u.s

   # determine pixel range
   p1 = cmin.closest_spectral_channel(v1)
   p2 = cmin.closest_spectral_channel(v2)

   for jj,ii in enumerate(range(p1, p2-1)):
       rgb = np.array([cmin[ii+2], cmin[ii+1], cmin[ii]]).T.swapaxes(0,1)

       # in case you manually set min/max
       rgb[rgb > max.value] = 1
       rgb[rgb < min.value] = 0

       # this is the unsupported little bit...
       F._ax1.clear()
       F._ax1.imshow((rgb-min.value)/(max-min).value, extent=F._extent)

       v1_ = int(np.round(cube.spectral_axis[ii].value))
       v2_ = int(np.round(cube.spectral_axis[ii+2].value))

       # then write out the files
       F.save('rgb/{2}_v{0}to{1}.png'.format(v1_, v2_, prefix))
       # make a sorted version for use with ffmpeg
       os.remove('rgb/{0:04d}.png'.format(jj))
       os.link('rgb/{2}_v{0}to{1}.png'.format(v1_, v2_, prefix), 'rgb/{0:04d}.png'.format(jj))

       print("Done with frame {1}: channel {0}".format(ii, jj))

   os.system('ffmpeg -y -i rgb/%04d.png -c:v libx264 -pix_fmt yuv420p -vf "scale=1024:768,setpts=10*PTS" -r 10 rgb/{0}_RGB_movie.mp4'.format(prefix))


3. Extract a beam-weighted spectrum from a cube


Each spectral cube has a 'beam' parameter if you have radio_beam
installed.  You can use that to create a beam kernel:

.. code:: python

    kernel = cube.beam.as_kernel(cube.wcs.pixel_scale_matrix[1,1])

Find the pixel you want to integrate over form the image.  e.g.,

.. code:: python
    x,y = 500, 150

Then, cut out an appropriate sub-cube and integrate over it

.. code-block:: python

    kernsize = kernel.shape[0]
    subcube = cube[:,y-kernsize/2.:y+kernsize/2., x-kernsize/2.:x+kernsize/2.]
    # create a boolean mask at the 1% of peak level (you can modify this)
    mask = kernel.array > (0.01*kernel.array.max())
    msubcube = subcube.with_mask(mask)
    # Then, take an appropriate beam weighting
    weighted_cube = msubcube * kernel.array
    # and *sum* (do not average!) over the weighted cube.
    beam_weighted_spectrum = weighted_cube.sum(axis=(1,2))
