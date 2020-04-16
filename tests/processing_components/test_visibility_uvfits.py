""" Unit tests for visibility operations
    
    
"""
import os
import sys
import unittest
import logging

import numpy
from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.visibility.base import create_blockvisibility_from_uvfits, create_visibility_from_uvfits
from rascil.processing_components.visibility.operations import integrate_visibility_by_channel
from rascil.processing_components.imaging.base import invert_2d, create_image_from_visibility
from rascil.processing_components.visibility.coalesce import convert_visibility_to_blockvisibility, \
    convert_blockvisibility_to_visibility
from rascil.processing_components.image.operations import export_image_to_fits


log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestCreateMS(unittest.TestCase):
    
    def setUp(self):
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", False)

        self.doplot = False
    
        return
    
    # def test_create_list(self):
    #     uvfitsfile = rascil_path("data/vis/xcasa.fits")
    #     self.vis = create_blockvisibility_from_uvfits(uvfitsfile)
    
    #     for v in self.vis:
    #         assert v.vis.data.shape[-1] == 4
    #         assert v.polarisation_frame.type == "circular"
    
    def test_create_list_spectral(self):
        
        uvfitsfile = rascil_path("data/vis/ASKAP_example.fits")
        
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_uvfits(uvfitsfile, range(schan, max_chan))
            vis_by_channel.append(v[0])
        
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"

    def test_create_list_spectral_average(self):
        
        uvfitsfile = rascil_path("data/vis/ASKAP_example.fits")
        
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_blockvisibility_from_uvfits(uvfitsfile, range(schan, max_chan))
            vis_by_channel.append(integrate_visibility_by_channel(v[0]))

        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.vis.data.shape[-2] == 1
            assert v.polarisation_frame.type == "linear"

    def test_invert(self):
        
        uvfitsfile = rascil_path("data/vis/ASKAP_example.fits")
        
        nchan_ave = 32
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            bv = create_blockvisibility_from_uvfits(uvfitsfile, range(schan, max_chan))[0]
            vis = convert_blockvisibility_to_visibility(bv)
            from rascil.processing_components.visibility.operations import convert_visibility_to_stokesI
            vis = convert_visibility_to_stokesI(vis)
            model = create_image_from_visibility(vis, npixel=256, polarisation_frame=PolarisationFrame('stokesI'))
            dirty, sumwt = invert_2d(vis, model, context='2d')
            assert (numpy.max(numpy.abs(dirty.data))) > 0.0
            assert dirty.shape == (nchan_ave, 1, 256, 256)
            if self.doplot:
                import matplotlib.pyplot as plt
                from rascil.processing_components.image.operations import show_image
                show_image(dirty)
                plt.show(block=False)
            if self.persist: export_image_to_fits(dirty, '%s/test_visibility_uvfits_dirty.fits' % self.dir)


if __name__ == '__main__':
    unittest.main()

