""" Functions to solve for and apply chains of antenna/station gain tables.

Calibration control is via a calibration_controls dictionary created by :py:func:`rascil.processing_components.calibration.chain_calibration.create_calibration_controls`. This supports the following Jones matrices::

   . T - Atmospheric phase
   . G - Electronics gain
   . P - Polarisation
   . B - Bandpass
   . I - Ionosphere

This is specified via a dictionary::

    contexts = {'T': {'shape': 'scalar', 'timeslice': 'auto', 'phase_only': True, 'first_iteration': 0},
                'G': {'shape': 'vector', 'timeslice': 60.0, 'phase_only': False, 'first_iteration': 0},
                'P': {'shape': 'matrix', 'timeslice': 1e4, 'phase_only': False, 'first_iteration': 0},
                'B': {'shape': 'vector', 'timeslice': 1e5, 'phase_only': False, 'first_iteration': 0},
                'I': {'shape': 'vector', 'timeslice': 1.0, 'phase_only': True, 'first_iteration': 0}}

Currently P and I are not supported.

For example::

    controls = create_calibration_controls()

    controls['T']['first_selfcal'] = 1
    controls['T']['phase_only'] = True
    controls['T']['timeslice'] = 'auto'

    controls['G']['first_selfcal'] = 3
    controls['G']['timeslice'] = 'auto'

    controls['B']['first_selfcal'] = 4
    controls['B']['timeslice'] = 1e5

    ical_list = ical_list_rsexecute_workflow(vis_list,
                                              model_imagelist=future_model_list,
                                              context='wstack', vis_slices=51,
                                              scales=[0, 3, 10], algorithm='mmclean',
                                              nmoment=3, niter=1000,
                                              fractional_threshold=0.1,
                                              threshold=0.1, nmajor=5, gain=0.25,
                                              deconvolve_facets=1,
                                              deconvolve_overlap=0,
                                              deconvolve_taper='tukey',
                                              timeslice='auto',
                                              psf_support=64,
                                              global_solution=False,
                                              calibration_context='TGB',
                                              do_selfcal=True)

"""

__all__ = ['calibrate_chain', 'solve_calibrate_chain', 'create_calibration_controls', 'apply_calibration_chain']

import logging

import numpy

from rascil.processing_components.visibility import convert_visibility_to_blockvisibility, convert_blockvisibility_to_visibility
from rascil.processing_components.calibration.operations import apply_gaintable, create_gaintable_from_blockvisibility, qa_gaintable
from rascil.data_models.memory_data_models import Visibility, BlockVisibility
from rascil.processing_components.calibration.solvers import solve_gaintable

log = logging.getLogger('logger')

def create_calibration_controls():
    """Create a dictionary containing default chanin calibration controls

    The fields are

        T: Atmospheric phase
        G: Electronic gains
        P: Polarisation
        B: Bandpass
        I: Ionosphere

    Therefore first get this default dictionary and then adjust parameters as desired. The calibrate function takes a context string e.g. TGB. It then calibrates each of these Jones matrices in turn

    Note that P and I calibration require off diagonal terms producing non-commutation of the Jones matrices. This is not handled yet.

    :return: dictionary
    """

    # controls = {'T': {'shape': 'scalar', 'timeslice': 'auto', 'phase_only': True, 'first_selfcal': 0},
    #             'G': {'shape': 'vector', 'timeslice': 60.0, 'phase_only': False, 'first_selfcal': 0},
    #             'P': {'shape': 'matrix', 'timeslice': 1e4, 'phase_only': False, 'first_selfcal': 0},
    #             'B': {'shape': 'vector', 'timeslice': 1e5, 'phase_only': False, 'first_selfcal': 0},
    #             'I': {'shape': 'vector', 'timeslice': 1.0, 'phase_only': True, 'first_selfcal': 0}}

    controls = {'T': {'shape': 'scalar', 'timeslice': 'auto', 'phase_only': True, 'first_selfcal': 0},
                'G': {'shape': 'vector', 'timeslice': 60.0, 'phase_only': False, 'first_selfcal': 0},
                'B': {'shape': 'vector', 'timeslice': 1e5, 'phase_only': False, 'first_selfcal': 0}}

    return controls


def apply_calibration_chain(vis, gaintables, calibration_context='T', controls=None, iteration=0, tol=1e-6,
                            **kwargs):
    """ Calibrate using algorithm specified by calibration_context and the calibration controls

    The context string can denote a sequence of calibrations e.g. TGB with different timescales.

    :param vis:
    :param model_vis:
    :param calibration_context: calibration contexts in order of correction e.g. 'TGB'
    :param control: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to the 'first_selfcal' field.
    :param kwargs:
    :return: Calibrated data_models, dict(gaintables)
    """

    if controls is None:
        controls = create_calibration_controls()

    # Check to see if changes are required
    changes = False
    for c in calibration_context:
        if (iteration >= controls[c]['first_selfcal']) and (c in gaintables.keys()):
            changes = True

    if changes:

        isVis = isinstance(vis, Visibility)
        if isVis:
            avis = convert_visibility_to_blockvisibility(vis)
        else:
            avis = vis

        assert isinstance(avis, BlockVisibility), avis

        for c in calibration_context:
            if iteration >= controls[c]['first_selfcal']:
                avis = apply_gaintable(avis, gaintables[c], timeslice=controls[c]['timeslice'])

        if isVis:
            return convert_blockvisibility_to_visibility(avis)
        else:
            return avis
    else:
        return vis


def calibrate_chain(vis, model_vis, gaintables=None, calibration_context='T', controls=None,
                    iteration=0, tol=1e-8, **kwargs):
    """ Calibrate using algorithm specified by calibration_context

    The context string can denote a sequence of calibrations e.g. TGB with different timescales.

    :param vis:
    :param model_vis:
    :param calibration_context: calibration contexts in order of correction e.g. 'TGB'
    :param controls: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to the 'first_selfcal' field.
    :param kwargs:
    :return: Calibrated data_models, dict(gaintables)
    """
    if controls is None:
        controls = create_calibration_controls()

    # Check to see if changes are required
    changes = False
    for c in calibration_context:
        if iteration >= controls[c]['first_selfcal']:
            changes = True

    if changes:

        isVis = isinstance(vis, Visibility)
        if isVis:
            avis = convert_visibility_to_blockvisibility(vis)
        else:
            avis = vis

        isMVis = isinstance(model_vis, Visibility)
        if isMVis:
            amvis = convert_visibility_to_blockvisibility(model_vis)
        else:
            amvis = model_vis

        assert isinstance(avis, BlockVisibility), avis

        if gaintables is None:
            gaintables = dict()
            
        for c in calibration_context:
            if iteration >= controls[c]['first_selfcal']:
                if c not in gaintables.keys():
                    log.info("Creating new {} gaintable".format(c))
                    gaintables[c] = \
                        create_gaintable_from_blockvisibility(avis,
                                                              timeslice=controls[c]['timeslice'])
                gaintables[c] = solve_gaintable(avis, amvis,
                                                gt=gaintables[c],
                                                timeslice=controls[c]['timeslice'],
                                                phase_only=controls[c]['phase_only'],
                                                crosspol=controls[c]['shape'] == 'matrix',
                                                tol=tol)
                log.debug('calibrate_chain: Jones matrix %s, iteration %d' % (c, iteration))
                log.debug(qa_gaintable(gaintables[c],
                                       context='Jones matrix %s, iteration %d' % (c, iteration)))
                avis = apply_gaintable(avis, gaintables[c], inverse=True, timeslice=controls[c]['timeslice'])
            else:
                log.debug('calibrate_chain: Jones matrix %s not solved, iteration %d' % (c, iteration))

        if isVis:
            return convert_blockvisibility_to_visibility(avis), gaintables
        else:
            return avis, gaintables
    else:
        return vis, gaintables


def solve_calibrate_chain(vis, model_vis, gaintables=None, calibration_context='T',
                          controls=None, iteration=0, tol=1e-6, **kwargs):
    """ Calibrate using algorithm specified by calibration_context

    The context string can denote a sequence of calibrations e.g. TGB with different timescales.

    :param vis:
    :param model_vis:
    :param calibration_context: calibration contexts in order of correction e.g. 'TGB'
    :param controls: controls dictionary, modified as necessary
    :param iteration: Iteration number to be compared to the 'first_selfcal' field.
    :param kwargs:
    :return: Calibrated data_models, dict(gaintables)
    """
    if controls is None:
        controls = create_calibration_controls()

    isVis = isinstance(vis, Visibility)
    if isVis:
        avis = convert_visibility_to_blockvisibility(vis)
    else:
        avis = vis

    isMVis = isinstance(model_vis, Visibility)
    if isMVis:
        amvis = convert_visibility_to_blockvisibility(model_vis)
    else:
        amvis = model_vis

    assert isinstance(avis, BlockVisibility), avis

    assert amvis.__repr__() != avis.__repr__(), "Vis and model vis are the same object: convert problem"

    # Always return a gain table, even if null
    
    if gaintables is None:
        gaintables = dict()
        
    for c in calibration_context:
        if c not in gaintables.keys():
            gaintables[c] = \
                create_gaintable_from_blockvisibility(avis,
                                                      timeslice=controls[c]['timeslice'])
        if iteration >= controls[c]['first_selfcal']:
            if numpy.max(numpy.abs(vis.flagged_weight)) > 0.0 and (amvis is None or numpy.max(numpy.abs(amvis.vis)) > 0.0):
                gaintables[c] = solve_gaintable(avis, amvis,
                                                gt=gaintables[c],
                                                timeslice=controls[c]['timeslice'],
                                                phase_only=controls[c]['phase_only'],
                                                crosspol=controls[c]['shape'] == 'matrix',
                                                tol=tol)
                log.debug('calibrate_chain: Jones matrix %s, iteration %d' % (c, iteration))
                log.debug(qa_gaintable(gaintables[c], context='Jones matrix %s, iteration %d' % (c, iteration)))
            else:
                log.debug('calibrate_chain: Jones matrix %s not solved, iteration %d' % (c, iteration))
        else:
            log.debug('calibrate_chain: Jones matrix %s not solved, iteration %d' % (c, iteration))

    return gaintables
