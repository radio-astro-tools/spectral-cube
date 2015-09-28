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
   
   # chop out the NaN borders
   cmin = cube.minimal_subcube()
   
   # Create the WCS template
   F = aplpy.FITSFigure(cmin[0].hdu)

   # decide on the velocity range
   v1 = 30*u.km/u.s
   v2 = 60*u.km/u.s

   # determine pixel range
   p1 = cmin.closest_spectral_pixel(v1)
   p2 = cmin.closest_spectral_pixel(v2)

   for ii in range(p1, p2-1):
       rgb = np.array([cmin[ii], cmin[ii+1], cmin[ii+1]]).T.swapaxes(0,1)

       # this is the unsupported little bit...
       F._ax1.imshow((rgb-min.value)/(max-min).value, extent=F._extent)

       v1_ = cube.spectral_axis[ii]
       v2_ = cube.spectral_axis[ii+2]

       # then write out the files
       F.save('rgb/HC3N_v{0}to{1}.png'.format(v1_, v2_))

       print("Done with channel {0}".format(ii))
