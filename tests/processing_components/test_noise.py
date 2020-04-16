"""Unit tests for testing support


"""

import logging
import sys
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.noise import addnoise_visibility
from rascil.processing_components.visibility.base import create_visibility, create_blockvisibility, copy_visibility

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestNoise(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.dir = rascil_path('test_results')
        
        self.frequency = numpy.linspace(0.8e8, 1.2e8, 5)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
    
    def test_addnoise_visibility(self):
        self.vis = create_visibility(self.config, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, integration_time=300.0, polarisation_frame=PolarisationFrame('stokesIQUV'),
                                     channel_bandwidth=self.channel_bandwidth)
        original = copy_visibility(self.vis)
        self.vis = addnoise_visibility(self.vis)
        actual = numpy.std(numpy.abs(self.vis.vis - original.vis))
        assert abs(actual - 0.000622776961225623) < 1e-4, actual
    
    def test_addnoise_blockvisibility(self):
        self.vis = create_blockvisibility(self.config, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame('stokesIQUV'),
                                          channel_bandwidth=self.channel_bandwidth)
        original = copy_visibility(self.vis)
        self.vis = addnoise_visibility(self.vis)
        actual = numpy.std(numpy.abs(self.vis.vis - original.vis))
        assert abs(actual - 0.0006225400451837758) < 1e-4, actual


if __name__ == '__main__':
    unittest.main()
