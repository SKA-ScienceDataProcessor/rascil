"""Unit tests for testing support


"""
import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from arl.data.data_models import BlockVisibility
from arl.util.graph_support import create_simulate_vis_graph, create_corrupt_vis_graph, \
    create_load_vis_graph, create_dump_vis_graph, create_predict_gleam_model_graph, create_gleam_model_graph
from arl.graphs.graphs import create_predict_wstack_graph
from arl.image.operations import qa_image

log = logging.getLogger(__name__)


class TestTestingDaskGraphSupport(unittest.TestCase):
    def setUp(self):
        self.dir = './test_results'
        os.makedirs(self.dir, exist_ok=True)
        
        self.frequency = numpy.linspace(1e8, 1.5e8, 3)
        self.channel_bandwidth = numpy.array([2.5e7, 2.5e7, 2.5e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        
    def test_create_simulate_vis_graph(self):
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        assert len(vis_graph_list) == len(self.frequency)
        vt = vis_graph_list[0].compute()
        assert type(vt) == BlockVisibility
        assert vt.nvis > 0

    def test_predict_gleam_model_graph(self):
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        predicted_vis_graph_list = create_predict_gleam_model_graph(vis_graph_list, vis_slices=11, npixel=256,
                                                   c_predict_graph=create_predict_wstack_graph)
        vt = predicted_vis_graph_list[0].compute()
        assert numpy.max(numpy.abs(vt.vis)) > 0.0
        assert type(vt) == BlockVisibility
        assert vt.nvis > 0

    def test_gleam_model_graph(self):
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        model_list = create_gleam_model_graph(vis_graph_list[0], npixel=256)
        qa= qa_image(model_list.compute())
        assert qa.data['max'] > 0.0
    
    def test_dump_load_graph(self):
        data_dir = './test_data'
        os.makedirs(data_dir, exist_ok=True)

        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        dump_graph = create_dump_vis_graph(vis_graph_list, name='test_data/imaging_dask')
        dump_graph.compute()
        load_graph = create_load_vis_graph(name='test_data/imaging_dask')
        assert len(load_graph) == len(vis_graph_list)

    def test_corrupt_vis_graph(self):
        vis_graph_list = create_simulate_vis_graph(frequency=self.frequency, channel_bandwidth=self.channel_bandwidth)
        vis_graph_list = create_predict_gleam_model_graph(vis_graph_list, vis_slices=11, npixel=256,
                                                   c_predict_graph=create_predict_wstack_graph)
        vt = vis_graph_list[0].compute()
        assert numpy.max(numpy.abs(vt.vis)) > 0.0

        corrupted_vis_graph_list = create_simulate_vis_graph(frequency=self.frequency,
                                                             channel_bandwidth=self.channel_bandwidth)
        corrupted_vis_graph_list = create_predict_gleam_model_graph(corrupted_vis_graph_list, vis_slices=11, npixel=256,
                                                   c_predict_graph=create_predict_wstack_graph)
        corrupted_vis_graph_list = create_corrupt_vis_graph(corrupted_vis_graph_list, phase_error=1.0)
        cvt = corrupted_vis_graph_list[0].compute()
        assert numpy.max(numpy.abs(cvt.vis)) > 0.0
        assert type(vt) == BlockVisibility
        assert vt.nvis > 0
        assert numpy.max(numpy.abs(cvt.vis-vt.vis)) > 0.0