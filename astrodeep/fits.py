from astropy.io import fits

def read_fits(path):
    """

    :param path: str, file-like or `pathlib.Path`
        File to be opened.
    :return: data: array, headers: fits headers in the file
    """
    hdul = fits.open(path)
    data = hdul[0].data
    headers = hdul[0].header
    return data, headers

def write_fits(path, data, headers):
    """
    Writes fits data to file

    :param path: str, file-like or `pathlib.Path`
            File to write to.  If a file object, must be opened in a
            writeable mode.

    :param data: array or DELAYED, optional
            The data in the HDU.

    :param headers: `~astropy.io.fits.Header`, optional
            The header to be used (as a template).  If ``header`` is `None`, a
            minimal header will be provided.
    """
    hdu = fits.PrimaryHDU(data=data, header=headers)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=True)