""" Pipeline functions. SDP standard pipelines expressed as functions.
"""

__all__ = ['ical_list_rsexecute_workflow',
           'continuum_imaging_list_rsexecute_workflow',
           'spectral_line_imaging_list_rsexecute_workflow']

import logging

from rascil.data_models.parameters import get_parameter
from rascil.processing_components.griddata import create_pswf_convolutionfunction
from rascil.processing_components.visibility import copy_visibility
from rascil.workflows.rsexecute.calibration.calibration_rsexecute import calibrate_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import invert_list_rsexecute_workflow, \
    residual_list_rsexecute_workflow, \
    predict_list_rsexecute_workflow, subtract_list_rsexecute_workflow, \
    restore_list_rsexecute_workflow, deconvolve_list_rsexecute_workflow

log = logging.getLogger('logger')


def ical_list_rsexecute_workflow(vis_list, model_imagelist, context, vis_slices=1, facets=1,
                                 gcfcf=None, calibration_context='TG', do_selfcal=True, **kwargs):
    """Create graph for ICAL pipeline

    :param vis_list: List of vis (or graph)
    :param model_imagelist:  list of models (or graph)
    :param context: imaging context e.g. '2d'
    :param vis_slices: Number of visibility slices (time or w)
    :param facets: Number of facets on each x,y axis
    :param calibration_context: Sequence of calibration steps e.g. TGB
    :param do_selfcal: Do the selfcalibration?
    :param kwargs: Parameters for functions in components
    :return:
    """
    
    gt_list = list()
    
    if gcfcf is None:
        gcfcf = [rsexecute.execute(create_pswf_convolutionfunction)(model_imagelist[0])]
    
    psf_imagelist = invert_list_rsexecute_workflow(vis_list, model_imagelist, dopsf=True, context=context,
                                                   vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    
    model_vislist = [rsexecute.execute(copy_visibility, nout=1)(v, zero=True) for v in vis_list]
    
    if do_selfcal:
        cal_vis_list = [rsexecute.execute(copy_visibility, nout=1)(v) for v in vis_list]
    else:
        cal_vis_list = vis_list
    
    if do_selfcal:
        # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
        # form the residual visibility, then make the residual image
        predicted_model_vislist = predict_list_rsexecute_workflow(model_vislist, model_imagelist,
                                                                  context=context, vis_slices=vis_slices,
                                                                  facets=facets,
                                                                  gcfcf=gcfcf, **kwargs)
        cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(cal_vis_list,
                                                                  predicted_model_vislist,
                                                                  gt_list,
                                                                  calibration_context=calibration_context, **kwargs)
        
        def zero_model_image(im):
            log.info("ical_list_rsexecute_workflow: setting initial model to zero after initial selfcal")
            im.data[...] = 0.0
            return im
        
        model_imagelist = [rsexecute.execute(zero_model_image, nout=1)(model) for model in model_imagelist]
        
        residual_imagelist = invert_list_rsexecute_workflow(cal_vis_list, model_imagelist,
                                                            context=context, dopsf=False,
                                                            vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                            iteration=0, **kwargs)
    
    else:
        # If we are not selfcalibrating it's much easier and we can avoid an unnecessary round of gather/scatter
        # for visibility partitioning such as timeslices and wstack.
        residual_imagelist = residual_list_rsexecute_workflow(cal_vis_list, model_imagelist, context=context,
                                                              vis_slices=vis_slices, facets=facets, gcfcf=gcfcf,
                                                              **kwargs)
    
    deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(residual_imagelist, psf_imagelist,
                                                                    model_imagelist,
                                                                    prefix='ical cycle 0',
                                                                    **kwargs)
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            if do_selfcal:
                model_vislist = predict_list_rsexecute_workflow(model_vislist, deconvolve_model_imagelist,
                                                                context=context, vis_slices=vis_slices,
                                                                facets=facets,
                                                                gcfcf=gcfcf, **kwargs)
                cal_vis_list = [rsexecute.execute(copy_visibility)(v) for v in vis_list]
                cal_vis_list, gt_list = calibrate_list_rsexecute_workflow(cal_vis_list,
                                                                          model_vislist,
                                                                          gt_list,
                                                                          calibration_context=calibration_context,
                                                                          iteration=cycle, **kwargs)
                residual_vislist = subtract_list_rsexecute_workflow(cal_vis_list, model_vislist)
                residual_imagelist = invert_list_rsexecute_workflow(residual_vislist, model_imagelist,
                                                                    context=context,
                                                                    vis_slices=vis_slices, facets=facets,
                                                                    gcfcf=gcfcf, **kwargs)
            else:
                residual_imagelist = residual_list_rsexecute_workflow(cal_vis_list, deconvolve_model_imagelist,
                                                                      context=context,
                                                                      vis_slices=vis_slices, facets=facets,
                                                                      gcfcf=gcfcf,
                                                                      **kwargs)
            
            prefix = "ical cycle %d" % (cycle + 1)
            deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(residual_imagelist, psf_imagelist,
                                                                            deconvolve_model_imagelist,
                                                                            prefix=prefix,
                                                                            **kwargs)
    residual_imagelist = residual_list_rsexecute_workflow(cal_vis_list, deconvolve_model_imagelist, context=context,
                                                          vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    restore_imagelist = restore_list_rsexecute_workflow(deconvolve_model_imagelist, psf_imagelist, residual_imagelist)
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist, gt_list)


def continuum_imaging_list_rsexecute_workflow(vis_list, model_imagelist, context, gcfcf=None,
                                              vis_slices=1, facets=1, **kwargs):
    """ Create graph for the continuum imaging pipeline.
    
    Same as ICAL but with no selfcal.
    
    :param vis_list: List of vis (or graph)
    :param model_imagelist: List of models (or graph)
    :param context: Imaging context
    :param kwargs: Parameters for functions in components
    :return:
    """
    if gcfcf is None:
        gcfcf = [rsexecute.execute(create_pswf_convolutionfunction)(model_imagelist[0])]
    
    psf_imagelist = invert_list_rsexecute_workflow(vis_list, model_imagelist, context=context, dopsf=True,
                                                   vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    
    residual_imagelist = residual_list_rsexecute_workflow(vis_list, model_imagelist, context=context, gcfcf=gcfcf,
                                                          vis_slices=vis_slices, facets=facets, **kwargs)
    
    deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(residual_imagelist, psf_imagelist,
                                                                    model_imagelist,
                                                                    prefix='cip cycle 0',
                                                                    **kwargs)
    nmajor = get_parameter(kwargs, "nmajor", 5)
    if nmajor > 1:
        for cycle in range(nmajor):
            prefix = "cip cycle %d" % (cycle + 1)
            residual_imagelist = residual_list_rsexecute_workflow(vis_list, deconvolve_model_imagelist,
                                                                  context=context, vis_slices=vis_slices,
                                                                  facets=facets,
                                                                  gcfcf=gcfcf, **kwargs)
            deconvolve_model_imagelist = deconvolve_list_rsexecute_workflow(residual_imagelist, psf_imagelist,
                                                                            deconvolve_model_imagelist,
                                                                            prefix=prefix,
                                                                            **kwargs)
    
    residual_imagelist = residual_list_rsexecute_workflow(vis_list, deconvolve_model_imagelist, context=context,
                                                          vis_slices=vis_slices, facets=facets, gcfcf=gcfcf, **kwargs)
    restore_imagelist = restore_list_rsexecute_workflow(deconvolve_model_imagelist, psf_imagelist, residual_imagelist)
    return (deconvolve_model_imagelist, residual_imagelist, restore_imagelist)


def spectral_line_imaging_list_rsexecute_workflow(vis_list, model_imagelist, context, continuum_model_imagelist=None,
                                                  vis_slices=1, facets=1, gcfcf=None, **kwargs):
    """Create graph for spectral line imaging pipeline

    Uses the continuum imaging rsexecute pipeline after subtraction of a continuum model
    
    :param vis_list: List of vis (or graph)
    :param model_imagelist: List of Spectral line model (or graph)
    :param continuum_model_imagelist: Continuum model list (or graph)
    :param context: Imaging context
    :param vis_slices: Number of visibility slices (time or w)
    :param facets: Number of facets on each x,y axis
    :param kwargs: Parameters for functions in components
    :return: list of (deconvolved model, residual, restored) or graph
    """
    if continuum_model_imagelist is not None:
        vis_list = predict_list_rsexecute_workflow(vis_list, continuum_model_imagelist, context=context, gcfcf=gcfcf,
                                                   vis_slices=vis_slices, facets=facets, **kwargs)
    
    return continuum_imaging_list_rsexecute_workflow(vis_list, model_imagelist, context=context, gcfcf=gcfcf,
                                                     vis_slices=vis_slices, facets=facets, **kwargs)
