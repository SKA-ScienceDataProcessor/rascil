""" Unit tests for coordinate calculations

"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from astropy.time import Time

from rascil.processing_components.visibility import create_blockvisibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.visibility_geometry import calculate_blockvisibility_azel, \
    calculate_blockvisibility_hourangles, calculate_blockvisibility_transit_time


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(-21600, +21600, 3600.0)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-65.0 * u.deg, frame='icrs', equinox='J2000')
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        self.bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                           channel_bandwidth=self.channel_bandwidth,
                                           phasecentre=self.phasecentre,
                                           weight=1.0)
    
    def test_azel(self):
        azel = calculate_blockvisibility_azel(self.bvis)
        print(azel)
        numpy.testing.assert_array_almost_equal(azel[0][0].deg, 152.523126)
        numpy.testing.assert_array_almost_equal(azel[1][0].deg, 24.163175)
    
    def test_hourangle(self):
        ha = calculate_blockvisibility_hourangles(self.bvis)
        numpy.testing.assert_array_almost_equal(ha.deg[0], -89.743265)
    
    def test_transit_time(self):
        transit_time = calculate_blockvisibility_transit_time(self.bvis)
        numpy.testing.assert_array_almost_equal(transit_time.value[0], 2458850.395812)


if __name__ == '__main__':
    unittest.main()
