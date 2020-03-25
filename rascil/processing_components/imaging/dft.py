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

__all__ = ['dft_skycomponent_visibility', 'idft_visibility_skycomponent']

import collections
import logging
from typing import List, Union

import numpy

from rascil.data_models.memory_data_models import Visibility, BlockVisibility, Skycomponent, assert_same_chan_pol
from rascil.data_models.polarisation import convert_pol_frame
from rascil.processing_components.imaging.imaging_params import get_frequency_map
from rascil.processing_components.skycomponent import copy_skycomponent
from rascil.processing_components.visibility.base import calculate_visibility_phasor, calculate_blockvisibility_phasor

log = logging.getLogger('logger')


def dft_skycomponent_visibility(vis: Union[Visibility, BlockVisibility], sc: Union[Skycomponent, List[Skycomponent]]) \
        -> Union[Visibility, BlockVisibility]:
    """DFT to get the visibility from a Skycomponent, for Visibility or BlockVisibility

    :param vis: Visibility or BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :return: Visibility or BlockVisibility
    """
    if sc is None:
        return vis

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    for comp in sc:

        assert_same_chan_pol(vis, comp)
        assert isinstance(comp, Skycomponent), comp
        flux = comp.flux
        if comp.polarisation_frame != vis.polarisation_frame:
            flux = convert_pol_frame(flux, comp.polarisation_frame, vis.polarisation_frame)

        if isinstance(vis, Visibility):

            _, im_nchan = list(get_frequency_map(vis, None))
            phasor = calculate_visibility_phasor(comp.direction, vis)
            for row in range(vis.nvis):
                ic = im_nchan[row]
                vis.data['vis'][row, :] += flux[ic, :] * phasor[row]

        elif isinstance(vis, BlockVisibility):

            phasor = calculate_blockvisibility_phasor(comp.direction, vis)
            vis.data['vis'] += flux * phasor

    return vis


def idft_visibility_skycomponent(vis: Union[Visibility, BlockVisibility],
                                 sc: Union[Skycomponent, List[Skycomponent]]) -> \
        ([Skycomponent, List[Skycomponent]], List[numpy.ndarray]):
    """Inverse DFT a Skycomponent from Visibility or BlockVisibility

    :param vis: Visibility or BlockVisibility
    :param sc: Skycomponent or list of SkyComponents
    :return: Skycomponent or list of SkyComponents, array of weights
    """
    if sc is None:
        return sc

    if not isinstance(sc, collections.abc.Iterable):
        sc = [sc]

    newsc = list()
    weights_list = list()

    for comp in sc:
        assert isinstance(comp, Skycomponent), comp
        assert_same_chan_pol(vis, comp)
        newcomp = copy_skycomponent(comp)

        if isinstance(vis, Visibility):

            flux = numpy.zeros_like(comp.flux, dtype='complex')
            weight = numpy.zeros_like(comp.flux, dtype='float')
            _, im_nchan = list(get_frequency_map(vis, None))
            phasor = numpy.conjugate(calculate_visibility_phasor(comp.direction, vis))
            fvwp = vis.flagged_weight * vis.flagged_vis * phasor
            fw = vis.flagged_weight
            for row in range(vis.nvis):
                ic = im_nchan[row]
                flux[ic, :] += fvwp[row, :]
                weight[ic, :] += fw[row, :]

        elif isinstance(vis, BlockVisibility):

            phasor = numpy.conjugate(calculate_blockvisibility_phasor(comp.direction, vis))
            flux = numpy.sum(vis.flagged_weight * vis.flagged_vis * phasor, axis=(0, 1, 2))
            weight = numpy.sum(vis.flagged_weight, axis=(0, 1, 2))

        flux[weight > 0.0] = flux[weight > 0.0] / weight[weight > 0.0]
        flux[weight <= 0.0] = 0.0
        if comp.polarisation_frame != vis.polarisation_frame:
            flux = convert_pol_frame(flux, vis.polarisation_frame, comp.polarisation_frame)

        newcomp.flux = flux

        newsc.append(newcomp)
        weights_list.append(weight)

    return newsc, weights_list


