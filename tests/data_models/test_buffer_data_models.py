""" Unit tests for data model helpers. The helpers facilitate persistence of data models
using HDF5


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.buffer_data_models import BufferImage, BufferBlockVisibility, BufferGainTable, BufferSkyModel, \
    BufferConvolutionFunction, BufferGridData, BufferPointingTable, BufferFlagTable
from rascil.data_models.memory_data_models import Skycomponent, SkyModel, BlockVisibility
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.griddata.operations import create_griddata_from_image
from rascil.processing_components.griddata import create_convolutionfunction_from_image
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility
from rascil.processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import simulate_gaintable, create_test_image
from rascil.processing_components.simulation.pointing import simulate_pointingtable
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components import create_blockvisibility, create_flagtable_from_blockvisibility


class TestBufferDataModelHelpers(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.dir = rascil_path('test_results/')
        
        self.midcore = create_named_configuration('MID', rmax=3000.0)
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 100.0)
        self.frequency = numpy.linspace(1.0e9, 1.1e9, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])
        
        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.comp = Skycomponent(direction=self.compabsdirection, frequency=self.frequency, flux=self.flux)
    
    def test_readwriteblockvisibility(self):
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0, meta={"RASCIL":0.0})
        self.vis = dft_skycomponent_visibility(self.vis, self.comp)
        
        config = {"buffer": {"directory": self.dir},
                  "vislist": {"name": "test_bufferblockvisibility.hdf", "data_model": "BlockVisibility"}}
        bdm = BufferBlockVisibility(config["buffer"], config["vislist"], self.vis)
        bdm.sync()
        new_bdm = BufferBlockVisibility(config["buffer"], config["vislist"])
        new_bdm.sync()
        newvis = bdm.memory_data_model
        
        assert isinstance(newvis, BlockVisibility)
        assert numpy.array_equal(newvis.frequency, self.vis.frequency)
        assert newvis.data.shape == self.vis.data.shape
        assert numpy.max(numpy.abs(self.vis.vis - newvis.vis)) < 1e-15
        assert numpy.max(numpy.abs(self.vis.uvw - newvis.uvw)) < 1e-15
        assert numpy.abs(newvis.configuration.location.x.value - self.vis.configuration.location.x.value) < 1e-15
        assert numpy.abs(newvis.configuration.location.y.value - self.vis.configuration.location.y.value) < 1e-15
        assert numpy.abs(newvis.configuration.location.z.value - self.vis.configuration.location.z.value) < 1e-15
        assert numpy.max(numpy.abs(newvis.configuration.xyz - self.vis.configuration.xyz)) < 1e-15
        assert newvis.meta == self.vis.meta

    def test_readwritegaintable(self):
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)
        gt = create_gaintable_from_blockvisibility(self.vis, timeslice='auto')
        gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.1)

        config = {"buffer": {"directory": self.dir},
                  "gaintable": {"name": "test_buffergaintable.hdf", "data_model": "GainTable"}}
        bdm = BufferGainTable(config["buffer"], config["gaintable"], gt)
        bdm.sync()
        new_bdm = BufferGainTable(config["buffer"], config["gaintable"])
        new_bdm.sync()
        newgt = bdm.memory_data_model

        assert gt.data.shape == newgt.data.shape
        assert numpy.max(numpy.abs(gt.gain - newgt.gain)) < 1e-15

    def test_readwriteflagtable(self):
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)
        ft = create_flagtable_from_blockvisibility(self.vis, timeslice='auto')

        config = {"buffer": {"directory": self.dir},
                  "flagtable": {"name": "test_bufferflagtable.hdf", "data_model": "FlagTable"}}
        bdm = BufferFlagTable(config["buffer"], config["flagtable"], ft)
        bdm.sync()
        new_bdm = BufferFlagTable(config["buffer"], config["flagtable"])
        new_bdm.sync()
        newft = bdm.memory_data_model

        assert ft.data.shape == newft.data.shape
        assert numpy.max(numpy.abs(ft.flags - newft.flags)) < 1e-15

    def test_readwritepointingtable(self):
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)
        pt = create_pointingtable_from_blockvisibility(self.vis, timeslice='auto')
        pt = simulate_pointingtable(pt, pointing_error=0.1)
        
        config = {"buffer": {"directory": self.dir},
                  "pointingtable": {"name": "test_bufferpointingtable.hdf", "data_model": "PointingTable"}}
        bdm = BufferPointingTable(config["buffer"], config["pointingtable"], pt)
        bdm.sync()
        new_bdm = BufferPointingTable(config["buffer"], config["pointingtable"])
        new_bdm.sync()
        newpt = bdm.memory_data_model
    
        assert pt.data.shape == newpt.data.shape
        assert numpy.max(numpy.abs(pt.pointing - newpt.pointing)) < 1e-15

    def test_readwriteskymodel(self):
        vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)
        gt = create_gaintable_from_blockvisibility(vis, timeslice='auto')
        gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.1)
        im = create_test_image()
        sm = SkyModel(components=[self.comp], image=im, gaintable=gt)

        config = {"buffer": {"directory": self.dir},
                  "skymodel": {"name": "test_bufferskymodel.hdf", "data_model": "SkyModel"}}
        bdm = BufferSkyModel(config["buffer"], config["skymodel"], sm)
        bdm.sync()
        new_bdm = BufferSkyModel(config["buffer"], config["skymodel"])
        new_bdm.sync()
        newsm = bdm.memory_data_model

        assert newsm.components[0].flux.shape == self.comp.flux.shape
        assert newsm.image.data.shape == im.data.shape
        assert newsm.gaintable.data.shape == gt.data.shape
        assert numpy.max(numpy.abs(newsm.image.data - im.data)) < 1e-15

    def test_readwriteskymodel_no_image(self):
        vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          polarisation_frame=PolarisationFrame("linear"),
                                          weight=1.0)
        gt = create_gaintable_from_blockvisibility(vis, timeslice='auto')
        gt = simulate_gaintable(gt, phase_error=1.0, amplitude_error=0.1)
        sm = SkyModel(components=[self.comp], gaintable=gt)

        config = {"buffer": {"directory": self.dir},
                  "skymodel": {"name": "test_bufferskymodel.hdf", "data_model": "SkyModel"}}
        bdm = BufferSkyModel(config["buffer"], config["skymodel"], sm)
        bdm.sync()
        new_bdm = BufferSkyModel(config["buffer"], config["skymodel"])
        new_bdm.sync()
        newsm = bdm.memory_data_model

        assert newsm.components[0].flux.shape == self.comp.flux.shape
        assert newsm.gaintable.data.shape == gt.data.shape


    def test_readwriteimage(self):
        im = create_test_image()

        config = {"buffer": {"directory": self.dir},
                  "image": {"name": "test_bufferimage.hdf", "data_model": "Image"}}
        bdm = BufferImage(config["buffer"], config["image"], im)
        bdm.sync()
        new_bdm = BufferImage(config["buffer"], config["image"])
        new_bdm.sync()
        newim = bdm.memory_data_model

        assert newim.data.shape == im.data.shape
        assert numpy.max(numpy.abs(im.data - newim.data)) < 1e-15

    def test_readwriteimage_assertion(self):
        im = create_test_image()

        with self.assertRaises(AssertionError):
            config = {"buffer": {"directory": self.dir},
                    "image": {"name": "test_bufferskyimage.hdf", "data_model": "Image"}}
            bdm = BufferImage(config["buffer"], config["image"], im)
            bdm.sync()
            new_bdm = BufferSkyModel(config["buffer"], config["image"])
            new_bdm.sync()
            newim = bdm.memory_data_model

    def test_readwritegriddata(self):
        im = create_test_image()
        gd = create_griddata_from_image(im, None)
        config = {"buffer": {"directory": self.dir},
                  "griddata": {"name": "test_buffergriddata.hdf", "data_model": "GridData"}}
        bdm = BufferGridData(config["buffer"], config["griddata"], gd)
        bdm.sync()
        new_bdm = BufferGridData(config["buffer"], config["griddata"])
        new_bdm.sync()
        newgd = bdm.memory_data_model

        assert newgd.data.shape == gd.data.shape
        assert numpy.max(numpy.abs(gd.data - newgd.data)) < 1e-15

    def test_readwriteconvolutionfunction(self):
        im = create_test_image()
        cf = create_convolutionfunction_from_image(im)
        config = {"buffer": {"directory": self.dir},
                  "convolutionfunction": {"name": "test_bufferconvolutionfunction.hdf", "data_model":
                      "ConvolutionFunction"}}
        bdm = BufferConvolutionFunction(config["buffer"], config["convolutionfunction"], cf)
        bdm.sync()
        new_bdm = BufferConvolutionFunction(config["buffer"], config["convolutionfunction"])
        new_bdm.sync()
        newcf = bdm.memory_data_model

        assert newcf.data.shape == cf.data.shape
        assert numpy.max(numpy.abs(cf.data - newcf.data)) < 1e-15



if __name__ == '__main__':
    unittest.main()
