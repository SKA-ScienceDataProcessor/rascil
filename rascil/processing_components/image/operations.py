""" Image operations visible to the Execution Framework as Components

"""

__all__ = ['add_image',
           'calculate_image_frequency_moments',
           'calculate_image_from_frequency_moments',
           'convert_polimage_to_stokes',
           'convert_stokes_to_polimage',
           'copy_image',
           'create_empty_image_like',
           'create_image',
           'create_image_from_array',
           'create_w_term_like',
           'create_window',
           'export_image_to_fits',
           'fft_image',
           'image_is_canonical',
           'import_image_from_fits',
           'pad_image',
           'polarisation_frame_from_wcs',
           'qa_image',
           'remove_continuum_image',
           'reproject_image',
           'show_components',
           'show_image',
           'smooth_image',
           "scale_and_rotate_image",
           "apply_voltage_pattern_to_image"]

import copy
import logging
import warnings

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from reproject import reproject_interp

from rascil.data_models.memory_data_models import Image, QA
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import PolarisationFrame, convert_stokes_to_linear, convert_stokes_to_circular, \
    convert_linear_to_stokes, convert_circular_to_stokes
from rascil.processing_components.calibration import apply_jones
from rascil.processing_components.fourier_transforms import w_beam, fft, ifft

warnings.simplefilter('ignore', FITSFixedWarning)
log = logging.getLogger('logger')


def image_is_canonical(im: Image):
    """ Is this Image canonical format?

    :param im:
    :return:
    """
    if im is None:
        return True
    
    canonical = True
    canonical = canonical and len(im.shape) == 4
    canonical = canonical and im.wcs.wcs.ctype[0] == 'RA---SIN' and im.wcs.wcs.ctype[1] == 'DEC--SIN'
    canonical = canonical and im.wcs.wcs.ctype[2] == 'STOKES'
    canonical = canonical and (im.wcs.wcs.ctype[3] == 'FREQ' or im.wcs.wcs.ctype[3] == "MOMENT")
    
    if not canonical:
        log.debug("image_is_canonical: Image is not canonical 4D image with axes RA---SIN, DEC--SIN, STOKES, FREQ")
    
    return canonical


def export_image_to_fits(im: Image, fitsfile: str = 'imaging.fits'):
    """ Write an image to fits
    
    :param im: Image
    :param fitsfile: Name of output fits file in storage
    :returns: None

    See also
        :py:func:`rascil.processing_components.image.operations.import_image_from_array`

    """
    assert isinstance(im, Image), im
    return fits.writeto(filename=fitsfile, data=im.data, header=im.wcs.to_header(), overwrite=True)


def import_image_from_fits(fitsfile: str) -> Image:
    """ Read an Image from fits
    
    :param fitsfile: FITS file in storage
    :return: Image

    See also
        :py:func:`rascil.processing_components.image.operations.export_image_to_array`


    """
    fim = Image()
    warnings.simplefilter('ignore', FITSFixedWarning)
    hdulist = fits.open(fitsfile)
    fim.data = hdulist[0].data
    fim.wcs = WCS(fitsfile)
    hdulist.close()
    
    if len(fim.data) == 2:
        fim.polarisation_frame = PolarisationFrame('stokesI')
    else:
        try:
            fim.polarisation_frame = polarisation_frame_from_wcs(fim.wcs, fim.data.shape)
            # FITS and RASCIL polarisation conventions differ
            new_data = fim.data.copy()
            new_data[:, 3] = fim.data[:, 1]
            new_data[:, 1] = fim.data[:, 2]
            new_data[:, 2] = fim.data[:, 3]
            fim.data = new_data
        
        except ValueError:
            fim.polarisation_frame = PolarisationFrame('stokesI')
    
    log.debug("import_image_from_fits: created %s image of shape %s, size %.3f (GB)" %
              (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    log.debug("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    
    assert isinstance(fim, Image)
    return fim


def reproject_image(im: Image, newwcs: WCS, shape=None) -> (Image, Image):
    """ Re-project an image to a new coordinate system
    
    Currently uses the reproject python package. This seems to have some features do be careful using this method.
    For timeslice imaging griddata is used.

    :param im: Image to be reprojected
    :param newwcs: New WCS
    :param shape: Desired shape
    :return: Reprojected Image, Footprint Image
    """
    
    assert isinstance(im, Image), im
    
    if image_is_canonical(im):
        nchan, npol, ny, nx = im.shape
        if im.data.dtype == 'complex':
            rep_real = numpy.zeros(shape, dtype='float')
            rep_imag = numpy.zeros(shape, dtype='float')
            foot = numpy.zeros(shape, dtype='float')
            for chan in range(nchan):
                for pol in range(npol):
                    rep_real[chan, pol], foot[chan, pol] = reproject_interp((im.data.real[chan, pol], im.wcs.sub(2)), newwcs.sub(2), shape[2:], order='bicubic')
                    rep_imag[chan, pol], foot[chan, pol] = reproject_interp((im.data.imag[chan, pol], im.wcs.sub(2)), newwcs.sub(2), shape[2:], order='bicubic')
            rep = rep_real + 1j * rep_imag
        else:
            rep = numpy.zeros(shape, dtype='float')
            foot = numpy.zeros(shape, dtype='float')
            for chan in range(nchan):
                for pol in range(npol):
                    rep[chan, pol], foot[chan, pol] = reproject_interp((im.data[chan, pol], im.wcs.sub(2)), newwcs.sub(2), shape[2:], order='bicubic')
        
        if numpy.sum(foot.data) < 1e-12:
            log.warning("reproject_image: no valid points in reprojection")
    elif len(im.shape)==2:
        if im.data.dtype == 'complex':
            rep_real, foot = reproject_interp((im.data.real, im.wcs), newwcs, shape, order='bicubic')
            rep_imag, foot = reproject_interp((im.data.imag, im.wcs), newwcs, shape, order='bicubic')
            rep = rep_real + 1j * rep_imag
        else:
            rep, foot = reproject_interp((im.data, im.wcs), newwcs, shape, order='bicubic')
    
        if numpy.sum(foot.data) < 1e-12:
            log.warning("reproject_image: no valid points in reprojection")

    else:
        raise ValueError("Cannot reproject image with shape {}".format(im.shape))
        
    return create_image_from_array(rep, newwcs, im.polarisation_frame), create_image_from_array(foot, newwcs,
                                                                                                im.polarisation_frame)


def add_image(im1: Image, im2: Image) -> Image:
    """ Add two images
    
    :param im1: Image
    :param im2: Image
    :return: Image
    """
    assert isinstance(im1, Image), im1
    assert image_is_canonical(im1)
    assert isinstance(im2, Image), im2
    assert image_is_canonical(im2)
    
    assert im1.polarisation_frame == im2.polarisation_frame
    
    return create_image_from_array(im1.data + im2.data, im1.wcs, im1.polarisation_frame)


def qa_image(im, context="") -> QA:
    """Assess the quality of an image

    QA is a standard set of statistics of an image; max, min, maxabs, rms, sum, medianabs, medianabsdevmedian, median

    :param im:
    :return: QA
    """
    assert isinstance(im, Image), im
    data = {'shape': str(im.data.shape),
            'max': numpy.max(im.data),
            'min': numpy.min(im.data),
            'maxabs': numpy.max(numpy.abs(im.data)),
            'rms': numpy.std(im.data),
            'sum': numpy.sum(im.data),
            'medianabs': numpy.median(numpy.abs(im.data)),
            'medianabsdevmedian': numpy.median(numpy.abs(im.data - numpy.median(im.data))),
            'median': numpy.median(im.data)}
    
    qa = QA(origin="qa_image", data=data, context=context)
    return qa


def show_image(im: Image, fig=None, title: str = '', pol=0, chan=0, cm='Greys', components=None,
               vmin=None, vmax=None, vscale=1.0):
    """ Show an Image with coordinates using matplotlib, optionally with components

    :param im: Image
    :param fig: Matplotlib figure
    :param title: String for title of plot
    :param pol: Polarisation to show (index)
    :param chan: Channel to show (index)
    :param components: Optional components to be overlaid
    :param vmin: Clip to this minimum
    :param vmax: Clip to this maximum
    :param vscale: scale max, min by this amount
    :return:
    """
    import matplotlib.pyplot as plt
    
    assert isinstance(im, Image), im
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=im.wcs.sub([1, 2]))
    
    if len(im.data.shape) == 4:
        data_array = numpy.real(im.data[chan, pol, :, :])
    else:
        data_array = numpy.real(im.data)
    
    if vmax is None:
        vmax = vscale * numpy.max(data_array)
    if vmin is None:
        vmin = vscale * numpy.min(data_array)
    
    cm = ax.imshow(data_array, origin='lower', cmap=cm, vmax=vmax, vmin=vmin)
    
    ax.set_xlabel(im.wcs.wcs.ctype[0])
    ax.set_ylabel(im.wcs.wcs.ctype[1])
    ax.set_title(title)
    
    fig.colorbar(cm, orientation='vertical', shrink=0.7)
    
    if components is not None:
        for sc in components:
            x, y = skycoord_to_pixel(sc.direction, im.wcs, 0, 'wcs')
            ax.plot(x, y, marker='+', color='red')
    
    return fig


def show_components(im, comps, npixels=128, fig=None, vmax=None, vmin=None, title=''):
    """ Show components against an image

    :param im:
    :param comps:
    :param npixels:
    :param fig:
    :return:
    """
    import matplotlib.pyplot as plt
    
    if vmax is None:
        vmax = numpy.max(im.data[0, 0, ...])
    if vmin is None:
        vmin = numpy.min(im.data[0, 0, ...])
    
    if not fig:
        fig = plt.figure()
    plt.clf()
    
    assert image_is_canonical(im)
    
    for isc, sc in enumerate(comps):
        newim = copy_image(im)
        plt.subplot(111, projection=newim.wcs.sub([1, 2]))
        centre = numpy.round(skycoord_to_pixel(sc.direction, newim.wcs, 1, 'wcs')).astype('int')
        newim.data = \
            newim.data[:, :, (centre[1] - npixels // 2):(centre[1] + npixels // 2),
            (centre[0] - npixels // 2):(centre[0] + npixels // 2)]
        newim.wcs.wcs.crpix[0] -= centre[0] - npixels // 2
        newim.wcs.wcs.crpix[1] -= centre[1] - npixels // 2
        plt.imshow(newim.data[0, 0, ...], origin='lower', cmap='Greys', vmax=vmax, vmin=vmin)
        x, y = skycoord_to_pixel(sc.direction, newim.wcs, 0, 'wcs')
        plt.plot(x, y, marker='+', color='red')
        plt.title('Name = %s, flux = %s' % (sc.name, sc.flux))
        plt.show()


def smooth_image(model: Image, width=1.0, normalise=True):
    """ Smooth an image with a 2D Gaussian kernel
    
    :param model: Image
    :param width: Kernel width in pixels
    :param normalise: Normalise kernel peak to unity
    
    """
    assert isinstance(model, Image), model
    assert image_is_canonical(model)
    
    from astropy.convolution.kernels import Gaussian2DKernel
    from astropy.convolution import convolve_fft
    
    kernel = Gaussian2DKernel(width)
    
    cmodel = create_empty_image_like(model)
    nchan, npol, _, _ = model.shape
    for pol in range(npol):
        for chan in range(nchan):
            cmodel.data[chan, pol, :, :] = convolve_fft(model.data[chan, pol, :, :], kernel,
                                                        normalize_kernel=False,
                                                        allow_huge=True)
    if normalise and isinstance(kernel, Gaussian2DKernel):
        cmodel.data *= 2 * numpy.pi * width ** 2
    
    return cmodel


def calculate_image_frequency_moments(im: Image, reference_frequency=None, nmoment=1) -> Image:
    """Calculate frequency weighted moments of an image cube

    The frequency moments are calculated using:

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k


    Note that the spectral axis is replaced by a MOMENT axis.
    
    For example, to find the moments and then reconstruct from just the moments::
    
        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoment=5)
        reconstructed_cube = calculate_image_from_frequency_moments(model_multichannel, moment_cube)

    :param im: Image cube
    :param reference_frequency: Reference frequency (default None uses average)
    :param nmoment: Number of moments to calculate
    :return: Moments image
    """
    assert isinstance(im, Image)
    assert image_is_canonical(im)
    
    assert nmoment > 0
    nchan, npol, ny, nx = im.shape
    channels = numpy.arange(nchan)
    freq = im.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    
    assert nmoment <= nchan, "Number of moments %d cannot exceed the number of channels %d" % (nmoment, nchan)
    
    if reference_frequency is None:
        reference_frequency = numpy.average(freq)
    log.debug("calculate_image_frequency_moments: Reference frequency = %.3f (MHz)" % (reference_frequency / 1e6))
    
    moment_data = numpy.zeros([nmoment, npol, ny, nx])
    
    for moment in range(nmoment):
        for chan in range(nchan):
            weight = numpy.power((freq[chan] - reference_frequency) / reference_frequency, moment)
            moment_data[moment, ...] += im.data[chan, ...] * weight
    
    moment_wcs = copy.deepcopy(im.wcs)
    moment_wcs.wcs.ctype[3] = 'MOMENT'
    moment_wcs.wcs.crval[3] = 0.0
    moment_wcs.wcs.crpix[3] = 1.0
    moment_wcs.wcs.cdelt[3] = 1.0
    moment_wcs.wcs.cunit[3] = ''
    
    return create_image_from_array(moment_data, moment_wcs, im.polarisation_frame)


def calculate_image_from_frequency_moments(im: Image, moment_image: Image, reference_frequency=None) -> Image:
    """Calculate channel image from frequency weighted moments

    .. math::

        w_k = \\left(\\left(\\nu - \\nu_{ref}\\right) /  \\nu_{ref}\\right)^k


    Note that a new image is created
    
    For example, to find the moments and then reconstruct from just the moments::
    
        moment_cube = calculate_image_frequency_moments(model_multichannel, nmoment=5)
        reconstructed_cube = calculate_image_from_frequency_moments(model_multichannel, moment_cube)


    :param im: Image cube to be reconstructed
    :param moment_image: Moment cube (constructed using calculate_image_frequency_moments)
    :param reference_frequency: Reference frequency (default None uses average)
    :return: reconstructed image
    """
    assert isinstance(im, Image)
    nchan, npol, ny, nx = im.shape
    nmoment, mnpol, mny, mnx = moment_image.shape
    assert nmoment > 0
    
    assert npol == mnpol
    assert ny == mny
    assert nx == mnx
    
    assert moment_image.wcs.wcs.ctype[3] == 'MOMENT', "Second image should be a moment image"
    
    channels = numpy.arange(nchan)
    freq = im.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    
    if reference_frequency is None:
        reference_frequency = numpy.average(freq)
    log.debug("calculate_image_from_frequency_moments: Reference frequency = %.3f (MHz)" % (reference_frequency))
    
    newim = copy_image(im)
    
    newim.data[...] = 0.0
    
    for moment in range(nmoment):
        for chan in range(nchan):
            weight = numpy.power((freq[chan] - reference_frequency) / reference_frequency, moment)
            newim.data[chan, ...] += moment_image.data[moment, ...] * weight
    
    assert image_is_canonical(newim)
    
    return newim


def remove_continuum_image(im: Image, degree=1, mask=None):
    """ Fit and remove continuum visibility in place
    
    Fit a polynomial in frequency of the specified degree where mask is True and remove it from the image

    :param im:
    :param degree: 1 is a constant, 2 is a slope, etc.
    :param mask: Frequency mask
    :return:
    """
    assert isinstance(im, Image)
    assert image_is_canonical(im)
    
    if mask is not None:
        assert numpy.sum(mask) > 2 * degree, "Insufficient channels for fit"
    
    nchan, npol, ny, nx = im.shape
    channels = numpy.arange(nchan)
    frequency = im.wcs.sub(['spectral']).wcs_pix2world(channels, 0)[0]
    frequency -= frequency[nchan // 2]
    frequency /= numpy.max(frequency)
    wt = numpy.ones_like(frequency)
    if mask is not None:
        wt[mask] = 0.0
    
    for pol in range(npol):
        for y in range(ny):
            for x in range(nx):
                fit = numpy.polyfit(frequency, im.data[:, pol, y, x], w=wt, deg=degree)
                prediction = numpy.polyval(fit, frequency)
                im.data[:, pol, y, x] -= prediction
    return im


def convert_stokes_to_polimage(im: Image, polarisation_frame: PolarisationFrame):
    """Convert a stokes image in IQUV to polarisation_frame

    For example::
        impol = convert_stokes_to_polimage(imIQUV, Polarisation_Frame('linear'))

    :param im: Image to be converted
    :param polarisation_frame: desired polarisation frame
    :returns: Complex image

    See also
        :py:func:`rascil.processing_components.image.operations.convert_polimage_to_stokes`
        :py:func:`rascil.data_models.polarisation.convert_circular_to_stokes`
        :py:func:`rascil.data_models.polarisation.convert_linear_to_stokes`
    """
    
    assert isinstance(im, Image)
    assert image_is_canonical(im)
    assert isinstance(polarisation_frame, PolarisationFrame)
    
    if polarisation_frame == PolarisationFrame('linear'):
        cimarr = convert_stokes_to_linear(im.data)
        return create_image_from_array(cimarr, im.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('linearnp'):
        cimarr = convert_stokes_to_linear(im.data)
        return create_image_from_array(cimarr, im.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('circular'):
        cimarr = convert_stokes_to_circular(im.data)
        return create_image_from_array(cimarr, im.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('circularnp'):
        cimarr = convert_stokes_to_circular(im.data)
        return create_image_from_array(cimarr, im.wcs, polarisation_frame)
    elif polarisation_frame == PolarisationFrame('stokesI'):
        return create_image_from_array(im.data.astype("complex"), im.wcs, PolarisationFrame('stokesI'))
    else:
        raise ValueError("Cannot convert stokes to %s" % (polarisation_frame.type))


def convert_polimage_to_stokes(im: Image):
    """Convert a polarisation image to stokes IQUV (complex)

    For example:
        imIQUV = convert_polimage_to_stokes(impol)

    :param im: Complex Image in linear or circular
    :returns: Complex image

    See also
        :py:func:`rascil.processing_components.image.operations.convert_stokes_to_polimage`
        :py:func:`rascil.data_models.polarisation.convert_stokes_to_circular`
        :py:func:`rascil.data_models.polarisation.convert_stokes_to_linear`

    """
    assert isinstance(im, Image)
    assert im.data.dtype == 'complex'
    
    if im.polarisation_frame == PolarisationFrame('linear'):
        cimarr = convert_linear_to_stokes(im.data)
        return create_image_from_array(numpy.real(cimarr), im.wcs, PolarisationFrame('stokesIQUV'))
    elif im.polarisation_frame == PolarisationFrame('linearnp'):
        cimarr = convert_linear_to_stokes(im.data)
        return create_image_from_array(numpy.real(cimarr), im.wcs, PolarisationFrame('stokesIQ'))
    elif im.polarisation_frame == PolarisationFrame('circular'):
        cimarr = convert_circular_to_stokes(im.data)
        return create_image_from_array(numpy.real(cimarr), im.wcs, PolarisationFrame('stokesIQUV'))
    elif im.polarisation_frame == PolarisationFrame('circularnp'):
        cimarr = convert_circular_to_stokes(im.data)
        return create_image_from_array(numpy.real(cimarr), im.wcs, PolarisationFrame('stokesIV'))
    elif im.polarisation_frame == PolarisationFrame('stokesI'):
        return create_image_from_array(numpy.real(im.data), im.wcs, PolarisationFrame('stokesI'))
    else:
        raise ValueError("Cannot convert %s to stokes" % (im.polarisation_frame.type))


def create_window(template, window_type, **kwargs):
    """Create a window image using one of a number of methods

    The window is 1.0 or 0.0

    window types:
        'quarter': Inner quarter of the image

        'no_edge': 'window_edge' pixels around edge set to zero

        'threshold': template image pixels < 'window_threshold' absolute value set to zero

    :param template: Template image
    :param window_type: 'quarter' | 'no_edge' | 'threshold'
    :return: New image containing window

    See also
        :py:func:`rascil.processing_components.image.deconvolution.deconvolve_cube`


    """
    
    assert image_is_canonical(template)
    
    window = create_empty_image_like(template)
    if window_type == 'quarter':
        qx = template.shape[3] // 4
        qy = template.shape[2] // 4
        window.data[..., (qy + 1):3 * qy, (qx + 1):3 * qx] = 1.0
        log.info('create_mask: Cleaning inner quarter of each sky plane')
    elif window_type == 'no_edge':
        edge = get_parameter(kwargs, 'window_edge', 16)
        nx = template.shape[3]
        ny = template.shape[2]
        window.data[..., (edge + 1):(ny - edge), (edge + 1):(nx - edge)] = 1.0
        log.info('create_mask: Window omits %d-pixel edge of each sky plane' % (edge))
    elif window_type == 'threshold':
        window_threshold = get_parameter(kwargs, 'window_threshold', None)
        if window_threshold is None:
            window_threshold = 10.0 * numpy.std(template.data)
        window.data[template.data >= window_threshold] = 1.0
        log.info('create_mask: Window omits all points below %g' % (window_threshold))
    elif window_type is None:
        log.info("create_mask: Mask covers entire image")
    else:
        raise ValueError("Window shape %s is not recognized" % window_type)
    
    return window


def image_sizeof(im: Image):
    """ Return size of Image in GB

    :param im: Image
    :return: Float, size in GB
    """
    return im.size()


# noinspection PyUnresolvedReferences
def create_image(npixel=512, cellsize=0.000015, polarisation_frame=PolarisationFrame("stokesI"),
                 frequency=numpy.array([1e8]), channel_bandwidth=numpy.array([1e6]),
                 phasecentre=None, nchan=None, dtype='float64') -> Image:
    """Create an empty template image consistent with the inputs.

    :param npixel: Number of pixels
    :param cellsize: cellsize in radians
    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param frequency: Array of frequencies (Hz)
    :param channel_bandwidth: Array of Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :param nchan: Number of channels in image
    :param dtype: Python data type for array
    :return: Image

    See also
        :py:func:`rascil.processing_components.image.operations.testing_support.create_image_from_array`
        :py:func:`rascil.processing_components.imaging.base.create_image_from_visibility`
        :py:func:`rascil.processing_components.simulation.create_test_image`
        :py:mod:`rascil.processing_components.simulation`

    """
    
    if phasecentre is None:
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
    
    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")
    
    npol = polarisation_frame.npol
    if nchan is None:
        nchan = len(frequency)
    
    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    w.naxis = 4
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    
    return create_image_from_array(numpy.zeros(shape, dtype=dtype), w, polarisation_frame=polarisation_frame)


def create_image_from_array(data: numpy.array, wcs: WCS, polarisation_frame: PolarisationFrame) -> Image:
    """ Create an image from an array and optional wcs

    The output image preserves a reference to the input array.

    :param data: Numpy.array
    :param wcs: World coordinate system
    :param polarisation_frame: Polarisation Frame
    :return: Image

    See also
        :py:func:`rascil.processing_components.image.operations.create_image`
        :py:func:`rascil.processing_components.imaging.base.create_image_from_visibility`

    """
    fim = Image()
    fim.polarisation_frame = polarisation_frame
    
    fim.data = data
    if wcs is None:
        fim.wcs = None
    else:
        fim.wcs = wcs.deepcopy()
    
    if image_sizeof(fim) >= 1.0:
        log.debug("create_image_from_array: created %s image of shape %s, size %.3f (GB)" %
                  (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    
    assert isinstance(fim, Image), "Type is %s" % type(fim)
    return fim


def polarisation_frame_from_wcs(wcs, shape) -> PolarisationFrame:
    """Convert wcs to polarisation_frame

    See FITS definition in Table 29 of https://fits.gsfc.nasa.gov/standard40/fits_standard40draft1.pdf
    or subsequent revision

        1 I Standard Stokes unpolarized
        2 Q Standard Stokes linear
        3 U Standard Stokes linear
        4 V Standard Stokes circular
        −1 RR Right-right circular
        −2 LL Left-left circular
        −3 RL Right-left cross-circular
        −4 LR Left-right cross-circular
        −5 XX X parallel linear
        −6 YY Y parallel linear
        −7 XY XY cross linear
        −8 YX YX cross linear

        stokesI [1]
        stokesIQUV [1,2,3,4]
        circular [-1,-2,-3,-4]
        linear [-5,-6,-7,-8]

    For example::
        pol_frame = polarisation_frame_from_wcs(im.wcs, im.shape)


    :param wcs: World Coordinate System
    :param shape: Shape corresponding to wcs
    :returns: Polarisation_Frame object
    """
    # The third axis should be stokes:
    
    polarisation_frame = None
    
    if len(shape) == 2:
        polarisation_frame = PolarisationFrame("stokesI")
    else:
        npol = shape[1]
        pol = wcs.sub(['stokes']).wcs_pix2world(range(npol), 0)[0]
        pol = numpy.array(pol, dtype='int')
        for key in PolarisationFrame.fits_codes.keys():
            keypol = numpy.array(PolarisationFrame.fits_codes[key])
            if numpy.array_equal(pol, keypol):
                polarisation_frame = PolarisationFrame(key)
                return polarisation_frame
    if polarisation_frame is None:
        raise ValueError("Cannot determine polarisation code")
    
    assert isinstance(polarisation_frame, PolarisationFrame)
    return polarisation_frame


def copy_image(im: Image):
    """ Copy an image

    Performs deepcopy of data_models, breaking reference semantics

    :param im:
    :return: Image
    """
    
    if im is None:
        return im
    
    assert isinstance(im, Image), im
    fim = Image()
    fim.polarisation_frame = im.polarisation_frame
    fim.data = copy.deepcopy(im.data)
    if im.wcs is None:
        fim.wcs = None
    else:
        fim.wcs = copy.deepcopy(im.wcs)
    if image_sizeof(fim) >= 1.0:
        log.debug("copy_image: copied %s image of shape %s, size %.3f (GB)" %
                  (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    assert type(fim) == Image
    return fim


def create_empty_image_like(im: Image) -> Image:
    """ Create an empty image like another in shape and wcs

    The data array is initialized to zero

    :param im:
    :return: Image

    See also
        :py:func:`rascil.processing_components.image.base.copy_image`
    """
    assert isinstance(im, Image), im
    fim = Image()
    fim.polarisation_frame = im.polarisation_frame
    fim.data = numpy.zeros_like(im.data)
    if im.wcs is None:
        fim.wcs = None
    else:
        fim.wcs = copy.deepcopy(im.wcs)
    if image_sizeof(im) >= 1.0:
        log.debug("create_empty_image_like: created %s image of shape %s, size %.3f (GB)" %
                  (fim.data.dtype, str(fim.shape), image_sizeof(fim)))
    assert isinstance(fim, Image), "Type is %s" % type(fim)
    return fim


def fft_image(im, template_image=None):
    """ WCS-aware FFT of a canonical image

    The only transforms supported are:
        RA--SIN, DEC--SIN <-> UU, VV
        XX, YY <-> KX, KY

    For example::

        from rascil.processing_components import create_test_image, fft_image
        im = create_test_image()
        print(im)
            Image:
                Shape: (1, 1, 256, 256)
                WCS: WCS Keywords
            Number of WCS axes: 4
            CTYPE : 'RA---SIN'  'DEC--SIN'  'STOKES'  'FREQ'
            CRVAL : 0.0  35.0  1.0  100000000.0
            CRPIX : 129.0  129.0  1.0  1.0
            PC1_1 PC1_2 PC1_3 PC1_4  : 1.0  0.0  0.0  0.0
            PC2_1 PC2_2 PC2_3 PC2_4  : 0.0  1.0  0.0  0.0
            PC3_1 PC3_2 PC3_3 PC3_4  : 0.0  0.0  1.0  0.0
            PC4_1 PC4_2 PC4_3 PC4_4  : 0.0  0.0  0.0  1.0
            CDELT : -0.000277777791  0.000277777791  1.0  100000.0
            NAXIS : 0  0
                Polarisation frame: stokesI
        print(fft_image(im))
            Image:
                Shape: (1, 1, 256, 256)
                WCS: WCS Keywords
            Number of WCS axes: 4
            CTYPE : 'UU'  'VV'  'STOKES'  'FREQ'
            CRVAL : 0.0  0.0  1.0  100000000.0
            CRPIX : 129.0  129.0  1.0  1.0
            PC1_1 PC1_2 PC1_3 PC1_4  : 1.0  0.0  0.0  0.0
            PC2_1 PC2_2 PC2_3 PC2_4  : 0.0  1.0  0.0  0.0
            PC3_1 PC3_2 PC3_3 PC3_4  : 0.0  0.0  1.0  0.0
            PC4_1 PC4_2 PC4_3 PC4_4  : 0.0  0.0  0.0  1.0
            CDELT : -805.7218610503596  805.7218610503596  1.0  100000.0
            NAXIS : 0  0
                Polarisation frame: stokesI

    :param im:
    :param template_image:
    :return:

    See also
        :py:func:`rascil.processing_components.fourier_transforms.fft_support.fft`
        :py:func:`rascil.processing_components.fourier_transforms.fft_support.ifft`
    """
    assert len(im.shape) == 4
    d2r = numpy.pi / 180.0
    ft_wcs = copy.deepcopy(im.wcs)
    ft_shape = im.shape
    if im.wcs.wcs.ctype[0] == 'RA---SIN' and im.wcs.wcs.ctype[1] == 'DEC--SIN':
        assert image_is_canonical(im)
        ft_wcs.wcs.axis_types[0] = 0
        ft_wcs.wcs.axis_types[1] = 0
        ft_wcs.wcs.crval[0] = 0.0
        ft_wcs.wcs.crval[1] = 0.0
        ft_wcs.wcs.crpix[0] = ft_shape[3] // 2 + 1
        ft_wcs.wcs.crpix[1] = ft_shape[2] // 2 + 1
        ft_wcs.wcs.ctype[0] = 'UU'
        ft_wcs.wcs.ctype[1] = 'VV'
        ft_wcs.wcs.cdelt[0] = 1.0 / (ft_shape[3] * d2r * im.wcs.wcs.cdelt[0])
        ft_wcs.wcs.cdelt[1] = 1.0 / (ft_shape[2] * d2r * im.wcs.wcs.cdelt[1])
        ft_data = ifft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
    elif im.wcs.wcs.ctype[0] == 'UU' and im.wcs.wcs.ctype[1] == 'VV':
        ft_wcs.wcs.crval[0] = template_image.wcs.wcs.crval[0]
        ft_wcs.wcs.crval[1] = template_image.wcs.wcs.crval[1]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[0]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[1]
        ft_wcs.wcs.ctype[0] = template_image.wcs.wcs.ctype[0]
        ft_wcs.wcs.ctype[1] = template_image.wcs.wcs.ctype[1]
        ft_wcs.wcs.cdelt[0] = template_image.wcs.wcs.cdelt[0]
        ft_wcs.wcs.cdelt[1] = template_image.wcs.wcs.cdelt[1]
        ft_data = fft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
    elif im.wcs.wcs.ctype[0] == 'XX' and im.wcs.wcs.ctype[1] == 'YY':
        ft_wcs.wcs.axis_types[0] = 0
        ft_wcs.wcs.axis_types[1] = 0
        ft_wcs.wcs.crval[0] = 0.0
        ft_wcs.wcs.crval[1] = 0.0
        ft_wcs.wcs.crpix[0] = ft_shape[3] // 2 + 1
        ft_wcs.wcs.crpix[1] = ft_shape[2] // 2 + 1
        ft_wcs.wcs.ctype[0] = 'KX'
        ft_wcs.wcs.ctype[1] = 'KY'
        ft_wcs.wcs.cdelt[0] = 1.0 / (ft_shape[3] * im.wcs.wcs.cdelt[0])
        ft_wcs.wcs.cdelt[1] = 1.0 / (ft_shape[2] * im.wcs.wcs.cdelt[1])
        ft_data = ifft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
    elif im.wcs.wcs.ctype[0] == 'KX' and im.wcs.wcs.ctype[1] == 'KY':
        ft_wcs.wcs.crval[0] = template_image.wcs.wcs.crval[0]
        ft_wcs.wcs.crval[1] = template_image.wcs.wcs.crval[1]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[0]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[1]
        ft_wcs.wcs.ctype[0] = template_image.wcs.wcs.ctype[0]
        ft_wcs.wcs.ctype[1] = template_image.wcs.wcs.ctype[1]
        ft_wcs.wcs.cdelt[0] = template_image.wcs.wcs.cdelt[0]
        ft_wcs.wcs.cdelt[1] = template_image.wcs.wcs.cdelt[1]
        ft_data = fft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
    elif im.wcs.wcs.ctype[0] == 'AZELGEO long' and im.wcs.wcs.ctype[1] == 'AZELGEO lati':
        ft_wcs.wcs.axis_types[0] = 0
        ft_wcs.wcs.axis_types[1] = 0
        ft_wcs.wcs.crval[0] = 0.0
        ft_wcs.wcs.crval[1] = 0.0
        ft_wcs.wcs.crpix[0] = ft_shape[3] // 2 + 1
        ft_wcs.wcs.crpix[1] = ft_shape[2] // 2 + 1
        ft_wcs.wcs.ctype[0] = 'UU_AZELGEO'
        ft_wcs.wcs.ctype[1] = 'VV_AZELGEO'
        ft_wcs.wcs.cdelt[0] = 1.0 / (ft_shape[3] * im.wcs.wcs.cdelt[0])
        ft_wcs.wcs.cdelt[1] = 1.0 / (ft_shape[2] * im.wcs.wcs.cdelt[1])
        ft_data = ifft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
    elif im.wcs.wcs.ctype[0] == 'UU_AZELGEO' and im.wcs.wcs.ctype[1] == 'VV_AZELGEO':
        ft_wcs.wcs.crval[0] = template_image.wcs.wcs.crval[0]
        ft_wcs.wcs.crval[1] = template_image.wcs.wcs.crval[1]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[0]
        ft_wcs.wcs.crpix[0] = template_image.wcs.wcs.crpix[1]
        ft_wcs.wcs.ctype[0] = template_image.wcs.wcs.ctype[0]
        ft_wcs.wcs.ctype[1] = template_image.wcs.wcs.ctype[1]
        ft_wcs.wcs.cdelt[0] = template_image.wcs.wcs.cdelt[0]
        ft_wcs.wcs.cdelt[1] = template_image.wcs.wcs.cdelt[1]
        ft_data = fft(im.data.astype('complex'))
        return create_image_from_array(ft_data, wcs=ft_wcs, polarisation_frame=im.polarisation_frame)
    
    
    else:
        raise NotImplementedError("Cannot FFT specified axes {0}, {1}".format(im.wcs.wcs.ctype[0], im.wcs.wcs.ctype[1]))


def pad_image(im: Image, shape):
    """Pad an image to desired shape, adding equally to all edges

    Appropriate for standard 4D image with axes (freq, pol, y, x). Only pads in y, x

    The wcs crpix is adjusted appropriately.

    :param im: Image to be padded
    :param shape: Shape in 4 dimensions
    :return: Padded image
    """
    
    if im.shape == shape:
        return im
    else:
        newwcs = copy.deepcopy(im.wcs)
        newwcs.wcs.crpix[0] = im.wcs.wcs.crpix[0] + shape[3] // 2 - im.shape[3] // 2
        newwcs.wcs.crpix[1] = im.wcs.wcs.crpix[1] + shape[2] // 2 - im.shape[2] // 2
        
        for axis, _ in enumerate(im.shape):
            if shape[axis] < im.shape[axis]:
                raise ValueError("Padded shape %s is smaller than input shape %s" % (shape, im.shape))
        
        newdata = numpy.zeros(shape, dtype=im.data.dtype)
        ystart = shape[2] // 2 - im.shape[2] // 2
        yend = ystart + im.shape[2]
        xstart = shape[3] // 2 - im.shape[3] // 2
        xend = xstart + im.shape[3]
        newdata[..., ystart:yend, xstart:xend] = im.data[...]
        return create_image_from_array(newdata, newwcs, polarisation_frame=im.polarisation_frame)


def create_w_term_like(im: Image, w, phasecentre=None, remove_shift=False, dopol=False) -> Image:
    """Create an image with a w term phase term in it:

    .. math::

        I(l,m) = e^{-2 \\pi j (w(\\sqrt{1-l^2-m^2}-1)}


    The phasecentre is used as the delay centre for the w term (i.e. where n==0)

    :param im: template image
    :param phasecentre: SkyCoord definition of phasecentre
    :param w: w value to evaluate
    :param remove_shift:
    :param dopol: Do screen in polarisation?
    :return: Image
    """
    
    assert image_is_canonical(im)
    fim_shape = list(im.shape)
    if not dopol:
        fim_shape[1] = 1
    
    fim_array = numpy.zeros(fim_shape, dtype='complex')
    fim = create_image_from_array(fim_array, wcs=im.wcs, polarisation_frame=im.polarisation_frame)
    
    cellsize = abs(fim.wcs.wcs.cdelt[0]) * numpy.pi / 180.0
    nchan, npol, _, npixel = fim_shape
    if phasecentre is SkyCoord:
        wcentre = phasecentre.to_pixel(im.wcs, origin=0)
    else:
        wcentre = [im.wcs.wcs.crpix[0] - 1.0, im.wcs.wcs.crpix[1] - 1.0]
    
    fim.data[:, :, ...] = w_beam(npixel, npixel * cellsize, w=w, cx=wcentre[0], cy=wcentre[1],
                                 remove_shift=remove_shift)
    
    fov = npixel * cellsize
    fresnel = numpy.abs(w) * (0.5 * fov) ** 2
    log.debug('create_w_term_image: For w = %.1f, field of view = %.6f, Fresnel number = %.2f' % (w, fov, fresnel))
    
    return fim


def scale_and_rotate_image(im, angle=0.0, scale=None, order=5):
    """ Scale and then rotate and image in x, y axes

    Applies scale then rotates

    :param im: Image
    :param angle: Angle in radians
    :param scale: Scale [scale_x, scale_y]
    :param order: Order of interpolation (0-5)
    :return:
    """
    from scipy.ndimage.interpolation import affine_transform
    
    nchan, npol, ny, nx = im.shape
    c_in = 0.5 * numpy.array([ny, nx])
    c_out = 0.5 * numpy.array([ny, nx])
    rot = numpy.array([[numpy.cos(angle), -numpy.sin(angle)],
                       [numpy.sin(angle), numpy.cos(angle)]])
    inv_rot = rot.T
    if scale is None:
        scale = [1.0, 1.0]
    
    newim = copy_image(im)
    inv_scale = numpy.diag(scale)
    inv_transform = numpy.dot(inv_scale, inv_rot)
    offset = c_in - numpy.dot(inv_transform, c_out)
    for chan in range(nchan):
        for pol in range(npol):
            if im.data.dtype == "complex":
                newim.data[chan, pol] = affine_transform(im.data[chan, pol].real,
                                                         inv_transform,
                                                         offset=offset,
                                                         order=order,
                                                         output_shape=(ny, nx)) + \
                                        1.0j * affine_transform(im.data[chan, pol].imag,
                                                                inv_transform,
                                                                offset=offset,
                                                                order=order,
                                                                output_shape=(ny, nx))
            elif im.data.dtype == "float":
                newim.data[chan, pol] = affine_transform(im.data[chan, pol].real,
                                                         inv_transform,
                                                         offset=offset,
                                                         order=order,
                                                         output_shape=(ny, nx))
            else:
                raise ValueError("Cannot process data type {}".format(im.data.dtype))
    
    return newim


def apply_voltage_pattern_to_image(im: Image, vp: Image, inverse=False, min_det=1e-1, **kwargs) -> Image:
    """Apply a voltage pattern to an image
    
    For each pixel, the application is as follows:
    
    I_{corrected}(l,m) = vp(l,m) I(l,m) jones(j,m).H

    :param im: Image to have jones applied
    :param vp: Jones image to be applied
    :param inverse: Apply the inverse (default=False)
    :param min_det: Minimum determinant to correct
    :return: new Image with Jones applied
    """
    
    assert image_is_canonical(im)
    
    assert isinstance(im, Image)
    assert isinstance(vp, Image)
    
    newim = create_empty_image_like(im)
    
    if inverse:
        log.debug('apply_gaintable: Apply inverse voltage pattern image')
    else:
        log.debug('apply_gaintable: Apply voltage pattern image')
    
    is_scalar = vp.shape[1] == 1

    nchan, npol, ny, nx = im.shape
    
    assert im.shape == vp.shape
    
    if is_scalar:
        log.debug('apply_voltage_pattern_to_image: Scalar voltage pattern')
        if inverse:
            for chan in range(nchan):
                pb = (vp.data[chan, 0, ...] * numpy.conjugate(vp.data[chan, 0, ...])).real
                newim.data[chan, 0, ...] *= pb
        else:
            for chan in range(nchan):
                pb = (vp.data[chan, 0, ...] * numpy.conjugate(vp.data[chan, 0, ...])).real
                mask = pb > 0.0
                newim.data[chan, 0, ...][mask] /= pb[mask]
    else:
        log.debug('apply_voltage_pattern_to_image: Full Jones voltage pattern')
        polim = convert_stokes_to_polimage(im, vp.polarisation_frame)
        assert npol == 4
        im_t = numpy.transpose(polim.data, (0, 2, 3, 1)).reshape([nchan, ny, nx, 2, 2])
        vp_t = numpy.transpose(vp.data, (0, 2, 3, 1)).reshape([nchan, ny, nx, 2, 2])
        newim_t = numpy.zeros([nchan, ny, nx, 2, 2], dtype='complex')
        for chan in range(nchan):
            for y in range(ny):
                for x in range(nx):
                    newim_t[chan, y, x] = apply_jones(vp_t[chan, y, x], im_t[chan, y, x], inverse, min_det=min_det)
        
        newim.data = newim_t.reshape([nchan, ny, nx, 4]).transpose((0, 3, 1, 2))
        newim.polarisation_frame = vp.polarisation_frame
        newim = convert_polimage_to_stokes(newim)
        
        return newim
