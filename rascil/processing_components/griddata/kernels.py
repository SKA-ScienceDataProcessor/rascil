"""
Functions that define and manipulate kernels

"""

__all__ = ['create_pswf_convolutionfunction', 'create_box_convolutionfunction',
           'create_awterm_convolutionfunction', 'create_vpterm_convolutionfunction']

import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import Image
from rascil.processing_components.fourier_transforms.fft_coordinates import coordinates, grdsf
from rascil.processing_components.griddata.convolution_functions import create_convolutionfunction_from_image
from rascil.processing_components.image.operations import create_image_from_array, copy_image, create_empty_image_like, \
    fft_image, pad_image, create_w_term_like
from rascil.processing_components.imaging.primary_beams import convert_azelvp_to_radec

log = logging.getLogger('logger')

def create_box_convolutionfunction(im, oversampling=1, support=1):
    """ Fill a box car function into a ConvolutionFunction

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    assert isinstance(im, Image)
    cf = create_convolutionfunction_from_image(im, oversampling=1, support=4)

    nchan, npol, _, _ = im.shape

    cf.data[...] = 0.0 + 0.0j
    cf.data[..., 2, 2] = 1.0 + 0.0j

    # Now calculate the griddata correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im.data.shape
    nu = numpy.abs(coordinates(nx))

    gcf1d = numpy.sinc(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf = 1.0 / gcf

    gcf_data = numpy.zeros_like(im.data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(gcf_data, cf.projection_wcs, im.polarisation_frame)

    return gcf_image, cf


def create_pswf_convolutionfunction(im, oversampling=127, support=8):
    """ Fill an Anti-Aliasing filter into a ConvolutionFunction

    Fill the Prolate Spheroidal Wave Function into a GriData with the specified oversampling. Only the inner
    non-zero part is retained

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    assert isinstance(im, Image), im
    if oversampling % 2 == 0:
        oversampling += 1
        log.info("Setting oversampling to next greatest odd number {}".format(oversampling))

    width = support - 2
    # Calculate the convolution kernel. We oversample in u,v space by the factor oversampling
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)

    kernel = numpy.zeros([oversampling, support])
    for grid in range(1, support-1):
        for subsample in range(oversampling):
            nu = ((grid - support // 2) - (subsample - oversampling // 2) / oversampling)
            kernel[subsample, grid] = grdsf([nu / (width // 2)])[1]

    kernel /= numpy.sum(numpy.real(kernel[oversampling // 2, :]))

    nchan, npol, _, _ = im.shape

    cf.data = numpy.zeros([nchan, npol, 1, oversampling, oversampling, support, support]).astype('complex')
    for y in range(oversampling):
        for x in range(oversampling):
            cf.data[:, :, 0, y, x, :, :] = numpy.outer(kernel[y, :], kernel[x, :])[numpy.newaxis, numpy.newaxis, ...]

    for y in range(oversampling):
        for x in range(oversampling):
            norm = numpy.sum(numpy.real(cf.data[:, :, 0, y, x, :, :]), axis=(-2,-1))[..., numpy.newaxis, numpy.newaxis]
            cf.data[:, :, 0, y, x, :, :] /= norm

    # Now calculate the griddata correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im.data.shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d = grdsf(nu)[0]
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]

    gcf_data = numpy.zeros_like(im.data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(gcf_data, cf.projection_wcs, im.polarisation_frame)

    return gcf_image, cf


def create_awterm_convolutionfunction(im, make_pb=None, nw=1, wstep=1e15, oversampling=9, support=8, use_aaf=True,
                                      maxsupport=512, pa=None, normalise=True):
    """ Fill AW projection kernel into a GridData.

    :param im: Image template
    :param make_pb: Function to make the primary beam model image (hint: use a partial)
    :param nw: Number of w planes
    :param wstep: Step in w (wavelengths)
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as GridData
    """
    if oversampling % 2 == 0:
        oversampling +=1
        log.info("Setting oversampling to next greatest odd number {}".format(oversampling))
        
    d2r = numpy.pi / 180.0
    
    # We only need the griddata correction function for the PSWF so we make
    # it for the shape of the image
    nchan, npol, ony, onx = im.data.shape
    
    assert isinstance(im, Image)
    # Calculate the template convolution kernel.
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)
    
    cf_shape = list(cf.data.shape)
    assert nw > 0, "Number of w planes must be greater than zero"
    cf_shape[2] = nw
    cf.data = numpy.zeros(cf_shape).astype('complex')
    
    cf.grid_wcs.wcs.crpix[4] = nw // 2 + 1.0
    cf.grid_wcs.wcs.cdelt[4] = wstep
    cf.grid_wcs.wcs.ctype[4] = 'WW'
    if numpy.abs(wstep) > 0.0:
        w_list = cf.grid_wcs.sub([5]).wcs_pix2world(range(nw), 0)[0]
    else:
        w_list = [0.0]
    
    assert isinstance(oversampling, int)
    assert oversampling > 0
    
    nx = max(maxsupport, 2 * oversampling * support)
    ny = max(maxsupport, 2 * oversampling * support)
    
    qnx = nx // oversampling
    qny = ny // oversampling
    
    cf.data[...] = 0.0
    
    subim = copy_image(im)
    ccell = onx * numpy.abs(d2r * subim.wcs.wcs.cdelt[0]) / qnx
    
    subim.data = numpy.zeros([nchan, npol, qny, qnx])
    subim.wcs.wcs.cdelt[0] = -ccell / d2r
    subim.wcs.wcs.cdelt[1] = +ccell / d2r
    subim.wcs.wcs.crpix[0] = qnx // 2 + 1.0
    subim.wcs.wcs.crpix[1] = qny // 2 + 1.0
    
    if use_aaf:
        this_pswf_gcf, _ = create_pswf_convolutionfunction(subim, oversampling=1, support=6)
        norm = 1.0 / this_pswf_gcf.data
    else:
        norm = 1.0
    
    if make_pb is not None:
        pb = make_pb(subim)
        
        if pa is not None:
            rpb = convert_azelvp_to_radec(pb, subim, pa)
        else:
            rpb = convert_azelvp_to_radec(pb, subim, 0.0)
        
        norm *= rpb.data
    
    # We might need to work with a larger image
    padded_shape = [nchan, npol, ny, nx]
    thisplane = copy_image(subim)
    thisplane.data = numpy.zeros(thisplane.shape, dtype='complex')
    for z, w in enumerate(w_list):
        thisplane.data[...] = 0.0 + 0.0j
        thisplane = create_w_term_like(thisplane, w, dopol=True)
        thisplane.data *= norm
        paddedplane = pad_image(thisplane, padded_shape)
        paddedplane = fft_image(paddedplane)
        
        ycen, xcen = ny // 2, nx // 2
        for y in range(oversampling):
            ybeg = y + ycen + (support * oversampling) // 2 - oversampling // 2
            yend = y + ycen - (support * oversampling) // 2 - oversampling // 2
            # vv = range(ybeg, yend, -oversampling)
            for x in range(oversampling):
                xbeg = x + xcen + (support * oversampling) // 2 - oversampling // 2
                xend = x + xcen - (support * oversampling) // 2 - oversampling // 2
                
                # uu = range(xbeg, xend, -oversampling)
                cf.data[..., z, y, x, :, :] = paddedplane.data[..., ybeg:yend:-oversampling, xbeg:xend:-oversampling]
                # for chan in range(nchan):
                #     for pol in range(npol):
                #         cf.data[chan, pol, z, y, x, :, :] = paddedplane.data[chan, pol, :, :][vv, :][:, uu]
    
    if normalise:
        norm = numpy.zeros([nchan, npol, oversampling, oversampling])
        for y in range(oversampling):
            for x in range(oversampling):
                # uu = range(xbeg, xend, -oversampling)
                norm[..., y, x] = numpy.sum(numpy.real(cf.data[:, :, 0, y, x, :, :]), axis=(-2, -1))
        for z, _ in enumerate(w_list):
            for y in range(oversampling):
                for x in range(oversampling):
                    cf.data[:, :, z, y, x] /= norm[..., y, x][..., numpy.newaxis, numpy.newaxis]
    cf.data = numpy.conjugate(cf.data)
    
    if use_aaf:
        pswf_gcf, _ = create_pswf_convolutionfunction(im, oversampling=1, support=6)
    else:
        pswf_gcf = create_empty_image_like(im)
        pswf_gcf.data[...] = 1.0
    
    return pswf_gcf, cf


def create_vpterm_convolutionfunction(im, make_vp=None, oversampling=8, support=6, use_aaf=False,
                                      maxsupport=512, pa=None, normalise=True):
    """ Fill voltage pattern kernel projection kernel into a GridData.
    
    The makes the convolution function for gridding polarised data with a voltage
    pattern.

    :param im: Image template
    :param make_vp: Function to make the voltage pattern model image (hint: use a partial)
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as GridData
    """
    if oversampling % 2 == 0:
        log.info("Setting oversampling to next greatest odd number {}".format(oversampling))
        oversampling += 1

    d2r = numpy.pi / 180.0
    
    # We only need the griddata correction function for the PSWF so we make
    # it for the shape of the image
    nchan, npol, ony, onx = im.data.shape
    
    assert isinstance(im, Image)
    # Calculate the template convolution kernel.
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)
    
    cf_shape = list(cf.data.shape)
    cf.data = numpy.zeros(cf_shape).astype('complex')
    
    assert isinstance(oversampling, int)
    assert oversampling > 0
    
    nx = max(maxsupport, 2 * oversampling * support)
    ny = max(maxsupport, 2 * oversampling * support)
    
    qnx = nx // oversampling
    qny = ny // oversampling
    
    cf.data[...] = 0.0
    
    subim = copy_image(im)
    ccell = onx * numpy.abs(d2r * subim.wcs.wcs.cdelt[0]) / qnx
    
    subim.data = numpy.zeros([nchan, npol, qny, qnx])
    subim.wcs.wcs.cdelt[0] = -ccell / d2r
    subim.wcs.wcs.cdelt[1] = +ccell / d2r
    subim.wcs.wcs.crpix[0] = qnx // 2 + 1.0
    subim.wcs.wcs.crpix[1] = qny // 2 + 1.0
    
    vp = make_vp(subim)
    
    if pa is not None:
        rvp = convert_azelvp_to_radec(vp, subim, pa)
    else:
        rvp = convert_azelvp_to_radec(vp, subim, 0.0)

    if use_aaf:
        this_pswf_gcf, _ = create_pswf_convolutionfunction(subim, oversampling=1, support=6)
        rvp.data /= this_pswf_gcf.data

    # We might need to work with a larger image
    padded_shape = [nchan, npol, ny, nx]
    paddedplane = pad_image(rvp, padded_shape)
    paddedplane = fft_image(paddedplane)
    
    ycen, xcen = ny // 2, nx // 2
    for y in range(oversampling):
        ybeg = y + ycen + (support * oversampling) // 2 - oversampling // 2
        yend = y + ycen - (support * oversampling) // 2 - oversampling // 2
        # vv = range(ybeg, yend, -oversampling)
        for x in range(oversampling):
            xbeg = x + xcen + (support * oversampling) // 2 - oversampling // 2
            xend = x + xcen - (support * oversampling) // 2 - oversampling // 2
            
            # uu = range(xbeg, xend, -oversampling)
            cf.data[..., 0, y, x, :, :] = \
                paddedplane.data[..., ybeg:yend:-oversampling, xbeg:xend:-oversampling]
    
    if normalise:
        cf.data /= numpy.sum(numpy.real(cf.data[0, 0, 0, oversampling // 2, oversampling // 2, :, :]))
    cf.data = numpy.conjugate(cf.data)
    
    if use_aaf:
        pswf_gcf, _ = create_pswf_convolutionfunction(im, oversampling=1, support=6)
    else:
        pswf_gcf = create_empty_image_like(im)
        pswf_gcf.data[...] = 1.0
    
    return pswf_gcf, cf


def convert_image_to_kernel(im: Image, oversampling, kernelwidth):
    """ Convert an image to a griddata kernel

    :param im: Image to be converted
    :param oversampling: Oversampling of Image spatially
    :param kernelwidth: Kernel width to be extracted
    :return: numpy.ndarray[nchan, npol, oversampling, oversampling, kernelwidth, kernelwidth]
    """
    naxis = len(im.shape)

    assert numpy.max(numpy.abs(im.data)) > 0.0, "Image is empty"

    nchan, npol, ny, nx = im.shape
    assert nx % oversampling == 0, "Oversampling must be even"
    assert ny % oversampling == 0, "Oversampling must be even"

    assert kernelwidth < nx and kernelwidth < ny, "Specified kernel width %d too large"

    assert im.wcs.wcs.ctype[0] == 'UU', 'Axis type %s inappropriate for construction of kernel' % im.wcs.wcs.ctype[0]
    assert im.wcs.wcs.ctype[1] == 'VV', 'Axis type %s inappropriate for construction of kernel' % im.wcs.wcs.ctype[1]
    newwcs = WCS(naxis=naxis + 2)
    for axis in range(2):
        newwcs.wcs.ctype[axis] = im.wcs.wcs.ctype[axis]
        newwcs.wcs.crpix[axis] = kernelwidth // 2
        newwcs.wcs.crval[axis] = 0.0
        newwcs.wcs.cdelt[axis] = im.wcs.wcs.cdelt[axis] * oversampling

        newwcs.wcs.ctype[axis + 2] = im.wcs.wcs.ctype[axis]
        newwcs.wcs.crpix[axis + 2] = oversampling // 2
        newwcs.wcs.crval[axis + 2] = 0.0
        newwcs.wcs.cdelt[axis + 2] = im.wcs.wcs.cdelt[axis]

        # Now do Stokes and Frequency
        newwcs.wcs.ctype[axis + 4] = im.wcs.wcs.ctype[axis + 2]
        newwcs.wcs.crpix[axis + 4] = im.wcs.wcs.crpix[axis + 2]
        newwcs.wcs.crval[axis + 4] = im.wcs.wcs.crval[axis + 2]
        newwcs.wcs.cdelt[axis + 4] = im.wcs.wcs.cdelt[axis + 2]

    newdata_shape = [nchan, npol, oversampling, oversampling, kernelwidth, kernelwidth]

    newdata = numpy.zeros(newdata_shape, dtype=im.data.dtype)

    assert oversampling * kernelwidth < ny
    assert oversampling * kernelwidth < nx

    ystart = ny // 2 - oversampling * kernelwidth // 2
    xstart = nx // 2 - oversampling * kernelwidth // 2
    yend = ny // 2 + oversampling * kernelwidth // 2
    xend = nx // 2 + oversampling * kernelwidth // 2
    for chan in range(nchan):
        for pol in range(npol):
            for y in range(oversampling):
                slicey = slice(yend + y, ystart + y, -oversampling)
                for x in range(oversampling):
                    slicex = slice(xend + x, xstart + x, -oversampling)
                    newdata[chan, pol, y, x, ...] = im.data[chan, pol, slicey, slicex]

    return create_image_from_array(newdata, newwcs, polarisation_frame=im.polarisation_frame)
