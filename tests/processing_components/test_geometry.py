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
        self.utc_time = Time(["2020-01-01T00:00:00"], format='isot', scale='utc')
    
    def test_azel(self):
        utc_times = Time(numpy.arange(0.0, 1.0, 0.1) + self.utc_time.mjd, format='mjd', scale='utc')
        azel = calculate_azel(self.location, utc_times, self.phasecentre)
        numpy.testing.assert_array_almost_equal(azel[0][0].deg, -114.187525)
        numpy.testing.assert_array_almost_equal(azel[1][0].deg, 57.652575)
        numpy.testing.assert_array_almost_equal(azel[0][-1].deg, -171.622607)
        numpy.testing.assert_array_almost_equal(azel[1][-1].deg, 81.464305)

    def test_hourangles(self):
        ha = calculate_hourangles(self.location, self.utc_time, self.phasecentre)
        numpy.testing.assert_array_almost_equal(ha[0].deg, 36.627673)

    def test_transit_time(self):
        transit_time = calculate_transit_time(self.location, self.utc_time, self.phasecentre)
        numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.9)

    def test_transit_time_below_horizon(self):
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=+80.0 * u.deg, frame='icrs', equinox='J2000')
        transit_time = calculate_transit_time(self.location, self.utc_time, self.phasecentre)
        numpy.testing.assert_array_almost_equal(transit_time.mjd, 58849.9)



if __name__ == '__main__':
    unittest.main()
