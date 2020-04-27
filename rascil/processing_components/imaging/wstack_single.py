"""
The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
approximated as:

.. math::

    V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.
"""

__all__ = ['predict_wstack_single', 'invert_wstack_single']

import logging

import numpy

from rascil.data_models.memory_data_models import Visibility, Image
from rascil.processing_components.image import copy_image, create_w_term_like, \
    image_is_canonical, convert_stokes_to_polimage, convert_polimage_to_stokes
from rascil.processing_components.griddata import grid_visibility_to_griddata, \
    degrid_visibility_from_griddata, fft_griddata_to_image, fft_image_to_griddata, \
    create_griddata_from_image
from rascil.processing_components.visibility.base import copy_visibility
from rascil.processing_components.imaging import normalize_sumwt, fill_vis_for_psf

log = logging.getLogger('logger')


def predict_wstack_single(vis, model, remove=True, gcfcf=None, **kwargs) -> Visibility:
    """ Predict using a single w slices.
    
    This processes a single w plane, rotating out the w beam for the average w

    The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
    approximated as:

    .. math::

        V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

    If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.

    :param vis: Visibility to be predicted
    :param model: model image
    :return: resulting visibility (in place works)
    """

    assert isinstance(vis, Visibility), "wstack requires Visibility format not BlockVisibility"
    assert image_is_canonical(model)

    vis.data['vis'][...] = 0.0

    log.debug("predict_wstack_single: predicting using single w slice")

    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(vis.w)
    if remove:
        vis.data['uvw'][..., 2] -= w_average
    tempvis = copy_visibility(vis)

    # Calculate w beam and apply to the model. The imaginary part is not needed
    workimage = convert_stokes_to_polimage(model, vis.polarisation_frame)
    w_beam = create_w_term_like(model, w_average, vis.phasecentre)
    workimage.data = numpy.conjugate(w_beam.data) * workimage.data

    gcf, cf = gcfcf

    griddata = create_griddata_from_image(model, vis)
    griddata = fft_image_to_griddata(workimage, griddata, gcf)
    vis = degrid_visibility_from_griddata(vis, griddata=griddata, cf=cf)

    if remove:
        vis.data['uvw'][..., 2] += w_average

    return vis


def invert_wstack_single(vis: Visibility, im: Image, dopsf, normalize=True, remove=True,
                         gcfcf=None, **kwargs) -> (Image, numpy.ndarray):
    """Process single w slice
    
    The w-stacking or w-slicing approach is to partition the visibility data by slices in w. The measurement equation is
    approximated as:

    .. math::

        V(u,v,w) =\\sum_i \\int \\frac{ I(l,m) e^{-2 \\pi j (w_i(\\sqrt{1-l^2-m^2}-1))})}{\\sqrt{1-l^2-m^2}} e^{-2 \\pi j (ul+vm)} dl dm

    If images constructed from slices in w are added after applying a w-dependent image plane correction, the w term will be corrected.

    :param vis: Visibility to be inverted
    :param im: image template (not changed)
    :param dopsf: Make the psf instead of the dirty image
    :param normalize: Normalize by the sum of weights (True)
    :returns: image, sum of weights
    """
    assert image_is_canonical(im)

    log.debug("invert_wstack_single: predicting using single w slice")
    
    kwargs['imaginary'] = True
    
    assert isinstance(vis, Visibility), "wstack requires Visibility format not BlockVisibility"
    
    if dopsf:
        vis = fill_vis_for_psf(vis)
    
    # We might want to do wprojection so we remove the average w
    w_average = numpy.average(vis.w)
    if remove:
        vis.data['uvw'][..., 2] -= w_average

    gcf, cf = gcfcf

    griddata = create_griddata_from_image(im, vis)
    griddata, sumwt = grid_visibility_to_griddata(vis, griddata=griddata, cf=cf)
    cim = fft_griddata_to_image(griddata, gcf)
    cim = normalize_sumwt(cim, sumwt)

    if remove:
        vis.data['uvw'][..., 2] += w_average

    # Calculate w beam and apply to the model. The imaginary part is not needed
    w_beam = create_w_term_like(im, w_average, vis.phasecentre)
    cworkimage = copy_image(cim)
#    cworkimage.data = numpy.conjugate(w_beam.data) * cim.data
    cworkimage.data = w_beam.data * cim.data
    workimage = convert_polimage_to_stokes(cworkimage)

    return workimage, sumwt
