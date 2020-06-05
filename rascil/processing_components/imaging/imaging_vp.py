"""
Functions that aid fourier transform processing. These are built on top of the core
functions in processing_components.fourier_transforms.

The measurement equation for a sufficently narrow field of view interferometer is:

.. math::

    V(u,v,w) =\\int I(l,m) e^{-2 \\pi j (ul+vm)} dl dm


The measurement equation for a wide field of view interferometer is:

.. math::

    V(u,v,w) =\\int \\frac{I(l,m)}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm + w(\\sqrt{1-l^2-m^2}-1))} dl dm

This and related modules contain various approachs for dealing with the wide-field problem where the
extra phase term in the Fourier transform cannot be ignored.
"""

__all__ = ['predict_vp', 'invert_vp']

import logging

import numpy

from rascil.data_models import BlockVisibility, Image, ConvolutionFunction, PolarisationFrame
from rascil.processing_components.griddata.gridding import grid_blockvisibility_pol_to_griddata, fft_griddata_to_image, \
    fft_image_to_griddata, degrid_blockvisibility_pol_from_griddata
from rascil.processing_components.griddata.operations import create_griddata_from_image
from rascil.processing_components.image import convert_polimage_to_stokes, \
    convert_stokes_to_polimage, apply_single_voltage_pattern_to_image, copy_image
from rascil.processing_components.imaging.base import shift_vis_to_image, normalize_sumwt, \
    fill_vis_for_psf
from rascil.processing_components.visibility.base import copy_visibility

log = logging.getLogger('logger')


def predict_vp(vis: BlockVisibility, model: Image, vp: Image, cf: ConvolutionFunction,
               **kwargs) -> BlockVisibility:
    """ Predict using convolutional degridding, full polarised version

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms of
    this function. Any shifting needed is performed here.

    :param vis: BlockVisibility to be predicted
    :param model: model image
    :param vp: Voltage pattern (Image)
    :param cf: Voltage pattern convolution function
    :param kwargs:
    :return: resulting visibility (in place works)
    """
    
    if model is None:
        return vis
    
    assert isinstance(vis, BlockVisibility), vis
    
    _, _, ny, nx = model.data.shape
    
    griddata = create_griddata_from_image(model, vis)
    polmodel = convert_stokes_to_polimage(model, vis.polarisation_frame)
    polmodel = apply_single_voltage_pattern_to_image(polmodel, vp)
    griddata = fft_image_to_griddata(polmodel, griddata)
    vis = degrid_blockvisibility_pol_from_griddata(vis, griddata=griddata, cf=cf)
    
    # Now we can shift the visibility from the image frame to the original visibility frame
    svis = shift_vis_to_image(vis, model, tangent=True, inverse=True)
    return svis


def invert_vp(vis: BlockVisibility, im: Image, vp: Image, cf: ConvolutionFunction, dopsf: bool = False,
              normalize: bool = True, grid_weights=False,
              **kwargs) -> (Image, numpy.ndarray):
    """ Invert using 2D convolution function, using the voltage pattern convolution function

    Use the image im as a template. Do PSF in a separate call.

    This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
    of this function. Any shifting needed is performed here.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :param vp: Voltage pattern (Image)
    :param cf: Voltage pattern convolution function
    :param grid_weights: Grid the weights at centre of the uv plane
    :param kwargs:
    :return: resulting image, sum of weights

    """
    assert isinstance(vis, BlockVisibility), vis
    
    svis = copy_visibility(vis)
    
    if dopsf:
        svis = fill_vis_for_psf(svis)
    
    svis = shift_vis_to_image(svis, im, tangent=True, inverse=False)
    griddata = create_griddata_from_image(im, svis)
    griddata, sumwt = grid_blockvisibility_pol_to_griddata(svis, griddata=griddata, cf=cf, grid_weights=grid_weights)
    cim = fft_griddata_to_image(griddata)
    cvp = copy_image(vp)
    cvp.data = numpy.conjugate(cvp.data)
    cim = apply_single_voltage_pattern_to_image(cim, cvp)
    if normalize:
        cim = normalize_sumwt(cim, sumwt)
    
    if grid_weights:
        cim.polarisation_frame = PolarisationFrame("stokesIQUV")
        return cim, sumwt
    else:
        return convert_polimage_to_stokes(cim, **kwargs), sumwt
