""" Visibility selectors for a BlockVisibility or Visibility.


"""
__all__ = ['vis_select_uvrange', 'vis_select_wrange']

import logging

import numpy

from rascil.data_models.memory_data_models import Visibility

log = logging.getLogger('logger')

def vis_select_uvrange(vis: Visibility, uvmin=0.0, uvmax=numpy.infty):
    """Return rows in valid region
    
    :param vis: Visibility
    :param uvmin:
    :param uvmax:
    :return: Boolean array of valid rows
    """

    assert isinstance(vis, Visibility)

    uvdist = numpy.sqrt(vis.u**2+vis.v**2)
    rows = (uvmin < uvdist) & (uvdist <= uvmax)
    return rows


def vis_select_wrange(vis: Visibility, wmax=numpy.infty):
    """Return rows in valid region

    :param vis: Visibility
    :param wmax: w max in wavelengths
    :return: Boolean array of valid rows
    """
    assert isinstance(vis, Visibility)

    absw = numpy.abs(vis.w)
    rows = (wmax >= absw)
    return rows
