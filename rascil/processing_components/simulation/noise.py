"""
Functions that add noise.

"""

__all__ = ['calculate_noise_blockvisibility', 'calculate_noise_visibility', 'addnoise_visibility']

import logging

import numpy

from rascil.data_models.memory_data_models import Visibility, BlockVisibility

log = logging.getLogger('logger')

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
