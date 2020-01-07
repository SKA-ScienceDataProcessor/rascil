"""
Functions that add noise.

"""

__all__ = ['calculate_noise_blockvisibility', 'calculate_noise_visibility', 'addnoise_visibility']

import csv
import logging
from typing import List

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from scipy import interpolate

from rascil.data_models.memory_data_models import Configuration, Image, GainTable, Skycomponent, SkyModel, PointingTable
from rascil.data_models.memory_data_models import Visibility, BlockVisibility
from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.calibration.calibration import create_calibration_controls
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from rascil.processing_components.image.operations import import_image_from_fits
from rascil.processing_components.imaging.base import predict_2d, predict_skycomponent_visibility, \
    create_image_from_visibility, advise_wide_field
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.skycomponent.operations import create_skycomponent, insert_skycomponent, \
    apply_beam_to_skycomponent, filter_skycomponents_by_flux
from rascil.processing_components.visibility.base import create_blockvisibility, create_visibility
from rascil.processing_components.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from rascil.processing_components.image.operations import create_image_from_array

log = logging.getLogger(__name__)

def calculate_noise_visibility(bandwidth, int_time, diameter, t_sys, eta):
    """Calculate noise rms per visibility

    :param bandwidth: (Hz)
    :param int_time: Integration time (s)
    :param diameter: Diameter (m)
    :param t_sys: (K)
    :param eta: Efficiency
    :returns: Sigma [nrows]
    """
    
    k_b = 1.38064852e-23
    area = numpy.pi * (diameter / 2.) ** 2
    bt = bandwidth * int_time
    sigma = (numpy.sqrt(2) * k_b * t_sys) / (area * eta * (numpy.sqrt(bt)))
    sigma *= 1e26
    return sigma


def calculate_noise_blockvisibility(bandwidth, int_time, diameter, t_sys, eta):
    """Calculate noise rms per visibility

    :param bandwidth: (Hz)
    :param int_time: Integration time (s)
    :param diameter: Diameter (m)
    :param t_sys: (K)
    :param eta: Efficiency
    :returns: Sigma [nrows, nchan]
    """
    
    k_b = 1.38064852e-23
    area = numpy.pi * (diameter / 2.) ** 2
    bt = numpy.outer(int_time, bandwidth)
    sigma = (numpy.sqrt(2) * k_b * t_sys) / (area * eta * (numpy.sqrt(bt)))
    sigma *= 1e26
    return sigma


def addnoise_visibility(vis, t_sys=None, eta=None):
    """ Add noise to a visibility
    
    TODO: Obtain sensitivity values from vis as a function of frequency
    
    :param vis:
    :param t_sys: System temperature
    :param eta: Efficiency
    :return: vis with noise added
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    if t_sys is None:
        t_sys = 20.0
    
    if eta is None:
        eta = 0.78
    
    # We need to handle Visibility and BlockVisibility separately since time and bandwidth are
    # stored differently
    if isinstance(vis, Visibility):
        sigma = calculate_noise_visibility(vis.data['channel_bandwidth'], vis.data['integration_time'],
                                           vis.configuration.diameter[0], t_sys=t_sys, eta=eta)
        log.debug('addnoise_visibility: RMS noise value: %g' % sigma[0])
        # Each pol gets a separate noise
        for pol in range(vis.npol):
            vis.data["vis"][:, pol].real += numpy.random.normal(0, sigma)
            vis.data["vis"][:, pol].imag += numpy.random.normal(0, sigma)
    elif isinstance(vis, BlockVisibility):
        sigma = calculate_noise_blockvisibility(vis.channel_bandwidth, vis.data['integration_time'],
                                                vis.configuration.diameter[0], t_sys=t_sys, eta=eta)
        log.debug('addnoise_visibility: RMS noise value (first integration, first channel): %g' % sigma[0, 0])
        for row in range(vis.nvis):
            for ant1 in range(vis.nants):
                for ant2 in range(ant1, vis.nants):
                    for pol in range(vis.npol):
                        vis.data["vis"][row, ant2, ant1, :, pol].real += numpy.random.normal(0, sigma[row, ...])
                        vis.data["vis"][row, ant2, ant1, :, pol].imag += numpy.random.normal(0, sigma[row, ...])
                        vis.data["vis"][row, ant1, ant2, :, pol] = \
                            numpy.conjugate(vis.data["vis"][row, ant2, ant1, :, pol])
    
    return vis
