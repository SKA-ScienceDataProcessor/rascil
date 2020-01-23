""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.imaging.base import predict_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components import create_flagtable_from_blockvisibility, qa_flagtable, \
    create_blockvisibility, create_flagtable_from_rows

class TestFlagTableOperations(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        self.polarisation_frame = PolarisationFrame("linear")

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux)
        
    def test_create_flagtable(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre,
                                     polarisation_frame=self.polarisation_frame,
                                     weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        print(ft)
        assert len(ft.data) == len(bvis.data)

    def test_create_flagtable_from_rows(self):
        bvis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                      polarisation_frame=self.polarisation_frame,
                                      phasecentre=self.phasecentre, weight=1.0)
        ft = create_flagtable_from_blockvisibility(bvis)
        rows = ft.time > 150.0
        ft = create_flagtable_from_blockvisibility(bvis)
        selected_ft = create_flagtable_from_rows(ft, rows)
        assert len(selected_ft.time) == numpy.sum(numpy.array(rows))

if __name__ == '__main__':
    unittest.main()
