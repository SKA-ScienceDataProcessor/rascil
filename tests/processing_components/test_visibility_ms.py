""" Unit tests for visibility operations


"""
import sys
import unittest
import logging

import numpy

from rascil.data_models import rascil_path, rascil_data_path, BlockVisibility
from rascil.processing_components.visibility.base import create_blockvisibility_from_ms, create_visibility_from_ms
from rascil.processing_components.visibility.operations import integrate_visibility_by_channel

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

class TestCreateMS(unittest.TestCase):
    
    def setUp(self):
    
        try:
            from casacore.tables import table  # pylint: disable=import-error
            self.casacore_available = True
#            except ModuleNotFoundError:
        except:
            self.casacore_available = False

    def test_create_list(self):
        
        if not self.casacore_available:
            return
        
        msfile = rascil_path("data/vis/xcasa.ms")
        self.vis = create_blockvisibility_from_ms(msfile)
        
        for v in self.vis:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "circular"

    def test_create_list_spectral(self):
        if not self.casacore_available:
            return
    
        msfile = rascil_path("data/vis/ASKAP_example.ms")
    
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(msfile, range(schan, max_chan))
            vis_by_channel.append(v[0])
    
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"

    def test_create_list_slice(self):
        if not self.casacore_available:
            return
    
        msfile = rascil_path("data/vis/ASKAP_example.ms")
    
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_blockvisibility_from_ms(msfile, start_chan=schan, end_chan=max_chan - 1)
            assert v[0].vis.shape[-2] == nchan_ave
            vis_by_channel.append(v[0])
    
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"

    def test_create_list_slice_visibility(self):
        if not self.casacore_available:
            return
    
        msfile = rascil_path("data/vis/ASKAP_example.ms")
    
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(msfile, start_chan=schan, end_chan=max_chan - 1)
            nchannels = len(numpy.unique(v[0].frequency))
            assert nchannels == nchan_ave
            vis_by_channel.append(v[0])
    
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"
            assert numpy.max(numpy.abs(v.vis)) > 0.0
            assert numpy.max(numpy.abs(v.flagged_vis)) > 0.0
            assert numpy.sum(v.weight) > 0.0
            assert numpy.sum(v.flagged_weight) > 0.0

    def test_create_list_average_slice_visibility(self):
        if not self.casacore_available:
            return
    
        msfile = rascil_path("data/vis/ASKAP_example.ms")
    
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(msfile, start_chan=schan, end_chan=max_chan - 1, average_channels=True)
            nchannels = len(numpy.unique(v[0].frequency))
            assert nchannels == 1
            vis_by_channel.append(v[0])
    
        assert len(vis_by_channel) == 12
        for ivis, v in enumerate(vis_by_channel):
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"
            assert numpy.max(numpy.abs(v.vis)) > 0.0, ivis
            assert numpy.max(numpy.abs(v.flagged_vis)) > 0.0, ivis
            assert numpy.sum(v.weight) > 0.0, ivis
            assert numpy.sum(v.flagged_weight) > 0.0, ivis

    def test_create_list_single(self):
        if not self.casacore_available:
            return
    
        msfile = rascil_path("data/vis/ASKAP_example.ms")
    
        vis_by_channel = list()
        nchan_ave = 1
        nchan = 8
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(msfile, start_chan=schan, end_chan=schan)
            vis_by_channel.append(v[0])
    
        assert len(vis_by_channel) == 8, len(vis_by_channel)
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"

    def test_create_list_spectral_average(self):
        if not self.casacore_available:
            return
    
        msfile = rascil_path("data/vis/ASKAP_example.ms")
    
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_blockvisibility_from_ms(msfile, range(schan, max_chan), average_channels=True)
            vis_by_channel.append(v[0])
    
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.vis.data.shape[-2] == 1
            assert v.polarisation_frame.type == "linear"
            assert numpy.max(numpy.abs(v.vis)) > 0.0
            assert numpy.max(numpy.abs(v.flagged_vis)) > 0.0
            
    def test_read_all(self):
        ms_list = ["vis/3C277.1C.16channels.ms", "vis/ASKAP_example.ms", "vis/sim-1.ms", "vis/sim-2.ms",
                   "vis/xcasa.ms"]
        
        for ms in ms_list:
            vis_list = create_blockvisibility_from_ms(rascil_data_path(ms))
            assert isinstance(vis_list[0], BlockVisibility)

    def test_read_not_ms(self):
    
        with self.assertRaises(RuntimeError):
            ms = "vis/ASKAP_example.fits"
            vis_list = create_blockvisibility_from_ms(rascil_data_path(ms))
            assert isinstance(vis_list[0], BlockVisibility)


if __name__ == '__main__':
    unittest.main()
