""" Unit tests for testing support


"""
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.memory_data_models import BlockVisibility
from rascil.workflows.rsexecute.execution_support.rsexecutebase import rsexecuteBase
from rascil.workflows.rsexecute.execution_support.rsexecute import get_dask_Client

from rascil.workflows.rsexecute.simulation.simulation_rsexecute import simulate_list_rsexecute_workflow

log = logging.getLogger(__name__)


class TestSimulationrsexecuteSupport(unittest.TestCase):
    def setUp(self):
        client = get_dask_Client(memory_limit=4 * 1024 * 1024 * 1024, n_workers=4, dashboard_address=None)
        global rsexecute
        rsexecute = rsexecuteBase(use_dask=True)
        rsexecute.set_client(client)

        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
    def tearDown(self):
        global rsexecute
        rsexecute.close()
        del rsexecute
        
    @unittest.skip("hanging in Jenkins")
    def test_create_simulate_vis_list(self):
        vis_list = simulate_list_rsexecute_workflow(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        assert len(vis_list) == len(self.frequency)
        vt = rsexecute.compute(vis_list[0], sync=True)
        assert isinstance(vt, BlockVisibility)
        assert vt.nvis > 0
 
