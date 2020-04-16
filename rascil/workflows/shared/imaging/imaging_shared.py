""" Imaging context definitions, potentially shared by other workflows

"""

__all__ = ['imaging_context', 'imaging_contexts', 'sum_invert_results', 'remove_sumwt', 'threshold_list',
           'sum_predict_results']

import numpy

import logging

from rascil.processing_components.image.operations import copy_image, create_empty_image_like

from rascil.processing_components.imaging import normalize_sumwt
from rascil.processing_components.visibility import copy_visibility
from rascil.processing_components.image import calculate_image_frequency_moments
from rascil.processing_components.imaging import predict_2d, invert_2d
from rascil.processing_components.visibility import  vis_null_iter, vis_timeslice_iter, vis_wslice_iter
from rascil.processing_components.imaging import  predict_timeslice_single, invert_timeslice_single
from rascil.processing_components.imaging import  predict_wstack_single, invert_wstack_single

log = logging.getLogger('logger')

def imaging_contexts():
    """Contains all the context information for imaging
    
    The fields are:
        predict: Predict function to be used
        invert: Invert function to be used
        image_iterator: Iterator for traversing images
        vis_iterator: Iterator for traversing visibilities
        inner: The innermost axis
    
    :return:
    """
    from rascil.processing_components.imaging.ng import predict_ng, invert_ng
    contexts = {'2d': {'predict': predict_2d,
                       'invert': invert_2d,
                       'vis_iterator': vis_null_iter},
                'ng': {'predict': predict_ng,
                       'invert': invert_ng,
                       'vis_iterator': vis_null_iter},
                'wprojection': {'predict': predict_2d,
                       'invert': invert_2d,
                       'vis_iterator': vis_null_iter},
                'wsnapshots': {'predict': predict_timeslice_single,
                       'invert': invert_timeslice_single,
                       'vis_iterator': vis_timeslice_iter},
                'facets': {'predict': predict_2d,
                           'invert': invert_2d,
                           'vis_iterator': vis_null_iter},
                'facets_ng': {'predict': predict_ng,
                           'invert': invert_ng,
                           'vis_iterator': vis_null_iter},
                'facets_timeslice': {'predict': predict_timeslice_single,
                                     'invert': invert_timeslice_single,
                                     'vis_iterator': vis_timeslice_iter},
                'facets_wstack': {'predict': predict_wstack_single,
                                  'invert': invert_wstack_single,
                                  'vis_iterator': vis_wslice_iter},
                'timeslice': {'predict': predict_timeslice_single,
                              'invert': invert_timeslice_single,
                              'vis_iterator': vis_timeslice_iter},
                'wstack': {'predict': predict_wstack_single,
                           'invert': invert_wstack_single,
                           'vis_iterator': vis_wslice_iter}}

    return contexts


def imaging_context(context='2d'):
    contexts = imaging_contexts()
    assert context in contexts.keys(), context
    return contexts[context]


def sum_invert_results_local(image_list):
    """ Sum a set of invert results with appropriate weighting
    without normalize_sumwt at the end
    :param image_list: List of [image, sum weights] pairs
    :return: image, sum of weights
    """
    
    first = True
    sumwt = 0.0
    im = None
    for i, arg in enumerate(image_list):
        if arg is not None:
            if isinstance(arg[1], numpy.ndarray):
                scale = arg[1][..., numpy.newaxis, numpy.newaxis]
            else:
                scale = arg[1]
            if first:
                im = copy_image(arg[0])
                im.data *= scale
                sumwt = arg[1].copy()
                first = False
            else:
                im.data += scale * arg[0].data
                sumwt += arg[1]
    
    assert not first, "No invert results"
    return im, sumwt

def sum_invert_results(image_list, normalize=True):
    """ Sum a set of invert results with appropriate weighting

    :param image_list: List of [image, sum weights] pairs
    :return: image, sum of weights
    """
    if len(image_list) == 1:
        return image_list[0]
    
    im = create_empty_image_like(image_list[0][0])
    sumwt = image_list[0][1].copy()
    sumwt *= 0.0

    for i, arg in enumerate(image_list):
        if arg is not None:
            im.data += arg[1][..., numpy.newaxis, numpy.newaxis] * arg[0].data
            sumwt += arg[1]

    if normalize:
        im = normalize_sumwt(im, sumwt)
    return im, sumwt


def remove_sumwt(results):
    """ Remove sumwt term in list of tuples (image, sumwt)

    :param results:
    :return: A list of just the dirty images
    """
    return [d[0] for d in results]


def sum_predict_results(results):
    """ Sum a set of predict results of the same shape

    :param results: List of visibilities to be summed
    :return: summed visibility
    """
    sum_results = None
    for result in results:
        if result is not None:
            if sum_results is None:
                sum_results = copy_visibility(result)
            else:
                assert sum_results.data['vis'].shape == result.data['vis'].shape
                sum_results.data['vis'] += result.data['vis']
    
    return sum_results


def threshold_list(imagelist, threshold, fractional_threshold, use_moment0=True, prefix=''):
    """ Find actual threshold for list of results, optionally using moment 0

    :param imagelist:
    :param threshold: Absolute threshold
    :param fractional_threshold: Fractional  threshold
    :param use_moment0: Use moment 0 for threshold
    :return:
    """
    peak = 0.0
    for i, result in enumerate(imagelist):
        if use_moment0:
            moments = calculate_image_frequency_moments(result)
            this_peak = numpy.max(numpy.abs(moments.data[0, ...] / result.shape[0]))
            peak = max(peak, this_peak)
            log.info("threshold_list: using moment 0, sub_image %d, peak = %f," % (i, this_peak))
        else:
            ref_chan = result.data.shape[0] // 2
            this_peak = numpy.max(numpy.abs(result.data[ref_chan]))
            peak = max(peak, this_peak)
            log.info("threshold_list: using refchan %d , sub_image %d, peak = %f," % (ref_chan, i, this_peak))

    actual = max(peak * fractional_threshold, threshold)
    
    
    if use_moment0:
        log.info("threshold_list %s: Global peak in moment 0 = %.6f, sub-image clean threshold will be %.6f" % (prefix,
                                                                                                           peak,
                                                                                                   actual))
    else:
        log.info("threshold_list %s: Global peak = %.6f, sub-image clean threshold will be %.6f" % (prefix, peak,
                                                                                                   actual))
    
    return actual
