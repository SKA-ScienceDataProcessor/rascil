"""Unit tests for testing support


"""
import os
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.imaging.base import create_image_from_visibility
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_visibility

from rascil.workflows.rsexecute.image.image_rsexecute import image_rsexecute_map_workflow
from rascil.processing_components.image.operations import export_image_to_fits
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger(__name__)


class TestImageGraph(unittest.TestCase):
    def setUp(self):
        rsexecute.set_client(verbose=False, memory_limit=4 * 1024 * 1024 * 1024, n_workers=4, dashboard_address=None)

        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.flux = numpy.array([[100.0], [100.0], [100.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration('LOWBD2-CORE')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
        
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def tearDown(self):
        rsexecute.close()

    def createVis(self, config, dec=-35.0, rmax=None):
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.config, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame('stokesI'))

    def test_map_create_pb(self):
        self.createVis(config='LOWBD2', rmax=1000.0)
        model = create_image_from_visibility(self.vis, cellsize=0.001, override_cellsize=False)
        beam = image_rsexecute_map_workflow(model, create_pb, facets=4, pointingcentre=self.phasecentre,
                                             telescope='MID')
        beam = rsexecute.compute(beam, sync=True)
        assert numpy.max(beam.data) > 0.0
        if self.persist: export_image_to_fits(beam, "%s/test_image_rsexecute_scatter_gather.fits" % (self.dir))
            
