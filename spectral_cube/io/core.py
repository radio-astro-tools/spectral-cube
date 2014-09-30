def read(filename, format=None, hdu=None, **kwargs):
    """
    Read a file into a :class:`SpectralCube` or :class:`StokesSpectralCube`
    instance.

    Parameters
    ----------
    filename : str or HDU
        File to read
    format : str, optional
        File format.
    hdu : int or str
        For FITS files, the HDU to read in (can be the ID or name of an
        HDU).
    kwargs : dict
        If the format is 'fits', the kwargs are passed to
        :func:`~astropy.io.fits.open`.

    Returns
    -------
    cube : :class:`SpectralCube` or :class:`StokesSpectralCube`
        The spectral cube read in
    """

    if format is None:
        format = determine_format_from_filename(filename)

    if format == 'fits':
        from .io.fits import load_fits_cube
        return load_fits_cube(filename)
    elif format == 'casa_image':
        from .io.casa_image import load_casa_image
        return load_casa_image(filename)
    elif format == 'class_lmv':
        from .io.class_lmv import load_lmv_cube
        return load_lmv_cube(filename)
    else:
        raise ValueError("Format {0} not implemented. Supported formats are "
                         "'fits', 'casa_image', and 'lmv'.".format(format))


def write(filename, cube, overwrite=False, format=None):
    """
    Write :class:`SpectralCube` or :class:`StokesSpectralCube` to a file.

    Parameters
    ----------
    filename : str
        Name of the output file
    cube : :class:`SpectralCube` or :class:`StokesSpectralCube`
        The spectral cube to write out
    overwrite : bool, optional
        Whether to overwrite the output file
    format : str, optional
        File format.
    """

    if format is None:
        format = determine_format(filename)

    if format == 'fits':
        from .fits import write_fits_cube
        write_fits_cube(filename, cube, overwrite=overwrite)
    else:
        raise ValueError("Format {0} not implemented. The only supported format is 'fits'".format(format))


def determine_format(input):

    from .fits import is_fits
    from .casa_image import is_casa_image

    if is_fits(input):
        return 'fits'
    elif is_casa_image(input):
        return 'casa_image'
    else:
        raise ValueError("Could not determine format - use the `format=` "
                         "parameter to explicitly set the format")
