"""

"""

__all__ = ['calibrate_list_serial_workflow']

from rascil.processing_components.calibration import apply_calibration_chain, solve_calibrate_chain
from rascil.processing_components.visibility import  convert_visibility_to_blockvisibility
from rascil.processing_components.visibility import visibility_gather_channel
from rascil.processing_components.visibility import integrate_visibility_by_channel, \
    divide_visibility


def calibrate_list_serial_workflow(vis_list, model_vislist, calibration_context='TG', global_solution=True,
                                       **kwargs):
    """ Create a set of components for (optionally global) calibration of a list of visibilities

    If global solution is true then visibilities are gathered to a single visibility data set which is then
    self-calibrated. The resulting gaintable is then effectively scattered out for application to each visibility
    set. If global solution is false then the solutions are performed locally.

    :param vis_list: list of visibilities
    :param model_vislist: list of model visibilities
    :param calibration_context: String giving terms to be calibrated e.g. 'TGB'
    :param global_solution: Solve for global gains
    :param kwargs: Parameters for functions in components
    :return: list of calibrated vis, list of dictionaries of gaintables
    """

    def solve(vis, modelvis=None):
        return solve_calibrate_chain(vis, modelvis, calibration_context=calibration_context, **kwargs)

    def apply(vis, gt):
        assert gt is not None
        return apply_calibration_chain(vis, gt, calibration_context=calibration_context, **kwargs)

    if global_solution and (len(vis_list) > 1):
        point_vislist = [convert_visibility_to_blockvisibility(v) for v in vis_list]
        point_modelvislist = [convert_visibility_to_blockvisibility(mv)
                              for mv in model_vislist]
        point_vislist = [divide_visibility(point_vislist[i], point_modelvislist[i])
                         for i, _ in enumerate(point_vislist)]
        global_point_vis_list = visibility_gather_channel(point_vislist)
        global_point_vis_list = integrate_visibility_by_channel(global_point_vis_list)
        # This is a global solution so we only compute one gain table
        gt_list = [solve(global_point_vis_list)]
        return [apply(v, gt_list[0]) for v in vis_list], gt_list
    else:
        gt_list = [solve(v, model_vislist[i])
                   for i, v in enumerate(vis_list)]
        return [apply(v, gt_list[i]) for i, v in enumerate(vis_list)], gt_list
