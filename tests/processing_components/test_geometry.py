""" Unit tests for coordinate calculations

"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

from rascil.processing_components.util.geometry import calculate_azel, calculate_hourangles, \
    calculate_transit_time


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.location = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
        self.times = (numpy.pi / 43200.0) * numpy.arange(-43200, +43200, 3600.0)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.utc_time = Time("2020-01-01T00:00:00", format='isot', scale='utc')
    
    def test_azel(self):
        azel = calculate_azel(self.location, self.utc_time, self.phasecentre)
        numpy.testing.assert_array_almost_equal(azel[0].deg, -113.96424106)
        numpy.testing.assert_array_almost_equal(azel[1].deg, 57.71575429)
    
    def test_hourangle(self):
        ha = calculate_hourangles(self.location, self.utc_time, self.phasecentre)
        numpy.testing.assert_array_almost_equal(ha.deg, 36.881315)
    
    def test_transit_time(self):
        transit_time = calculate_transit_time(self.location, self.utc_time, self.phasecentre)
        numpy.testing.assert_array_almost_equal(transit_time.value, 2458850.3958662455)


if __name__ == '__main__':
    unittest.main()
