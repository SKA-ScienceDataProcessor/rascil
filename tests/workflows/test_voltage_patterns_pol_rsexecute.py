"""Unit tests for voltage pattern application with polarisation

"""

import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import copy_image, \
    calculate_blockvisibility_parallactic_angles, convert_azelvp_to_radec, \
    simulate_gaintable_from_voltage_pattern
from rascil.processing_components.image import export_image_to_fits
from rascil.processing_components.image.operations import qa_image
from rascil.processing_components.imaging import create_image_from_visibility
from rascil.processing_components.imaging.primary_beams import create_vp
from rascil.processing_components.simulation import create_named_configuration
from rascil.workflows import calculate_residual_dft_rsexecute_workflow, \
    calculate_residual_fft_rsexecute_workflow, create_standard_mid_simulation_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger('logger')

log.setLevel(logging.DEBUG)


class TestVoltagePatternsPolGraph(unittest.TestCase):
    def setUp(self):
        rsexecute.set_client(use_dask=False)
        
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        self.persist = os.getenv("RASCIL_PERSIST", True)
    
    def tearDown(self):
        rsexecute.close()
    
    def createVis(self, config='MID', dec=-35.0, rmax=1e2, freq=1.3e9):
        self.frequency = numpy.array([freq])
        self.channel_bandwidth = numpy.array([1e6])
        self.flux = numpy.array([[100.0, 60.0, -10.0, +1.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
        self.config = create_named_configuration(config, rmax=rmax)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        self.npixel = 512
        self.fov = 4
        self.cellsize = numpy.pi * self.fov / (self.npixel * 180.0)
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
    
    def _test(self, flux=None, test_vp=False, method="dft", name=""):
        self.createVis(rmax=1e3)
        # Set up details of simulated observation
        
        npixel = 1024
        band = 'B2'
        frequency = [1.36e9]
        rmax = 1e4
        time_range = [-4.0, 4.0]
        #time_range = [-0.01, 0.01]
        time_chunk = 1800.0
        integration_time = 1800.0
        imaging_context = "2d"
        cellsize = 5e-06
        vis_slices = 1
        telescope = "MID_FEKO_B2"
        
        bvis_graph = create_standard_mid_simulation_rsexecute_workflow(band, rmax, self.phasecentre, time_range,
                                                                       time_chunk,
                                                                       integration_time,
                                                                       polarisation_frame=PolarisationFrame("linear"),
                                                                       zerow=imaging_context == "2d")
        bvis_graph = rsexecute.persist(bvis_graph)
        
        def find_vp_actual(telescope, normalise=True):
            vp = create_vp(telescope=telescope)
            if test_vp:
                vp.data[:, 0, ...] = 1.0
                vp.data[:, 1, ...] = 0.0
                vp.data[:, 2, ...] = 0.0
                vp.data[:, 3, ...] = 1.0
            if normalise:
                g = numpy.zeros([4])
                g[0] = numpy.max(numpy.abs(vp.data[:, 0, ...]))
                g[3] = numpy.max(numpy.abs(vp.data[:, 3, ...]))
                g[1] = g[2] = numpy.sqrt(g[0] * g[3])
                for chan in range(4):
                    vp.data[:, chan, ...] /= g[chan]
            return vp
        
        if flux is None:
            flux = [[1.0, 0.0, 0.0, 0.0]]
        else:
            flux = [flux]
        
        original_components = [Skycomponent(direction=self.phasecentre, frequency=frequency, flux=flux,
                                            polarisation_frame=PolarisationFrame('stokesIQUV'))]
        
        future_model_list = [
            rsexecute.execute(create_image_from_visibility)(bvis, npixel=npixel,
                                                            frequency=frequency,
                                                            nchan=1, cellsize=cellsize,
                                                            phasecentre=self.phasecentre,
                                                            polarisation_frame=PolarisationFrame("stokesIQUV"))
            for bvis in bvis_graph]
        
        centre_model = \
            [rsexecute.execute(create_image_from_visibility)(v, npixel=npixel,
                                                             nchan=1,
                                                             cellsize=cellsize,
                                                             phasecentre=self.phasecentre,
                                                             polarisation_frame=PolarisationFrame("stokesIQUV"))
             for v in bvis_graph]
        centre_model = rsexecute.persist(centre_model)
        
        # Now make all the residual images:
        result = dict()
        if method == "dft":
            # The parallactic angle rotation is done when the voltage pattern is
            # converted to a gaintable
            def make_ejterm(model):
                vp = find_vp_actual(telescope=telescope)
                return vp
            vp_list = [rsexecute.execute(make_ejterm)(centre_model[ibvis])
                       for ibvis, bvis in enumerate(bvis_graph)]
            vp_list = rsexecute.persist(vp_list)
            gt_list = [rsexecute.execute(simulate_gaintable_from_voltage_pattern)
                       (bvis, original_components, vp_list[ibv], use_radec=False)
                       for ibv, bvis in enumerate(bvis_graph)]
            gt_list = rsexecute.persist(gt_list)
            dirty_list = \
                calculate_residual_dft_rsexecute_workflow(bvis_graph, original_components,
                                                          future_model_list,
                                                          gt_list=gt_list,
                                                          context=imaging_context,
                                                          vis_slices=vis_slices,
                                                          do_wstacking=False)
            dirty_list = rsexecute.persist(dirty_list)
            
        else:
            def make_ejterm_rotated(model, bvis):
                vp = find_vp_actual(telescope=telescope)
                pa = numpy.average(calculate_blockvisibility_parallactic_angles(bvis))
                vp_rotated = convert_azelvp_to_radec(vp, model, pa)
                return vp_rotated
            
            vp_list = [rsexecute.execute(make_ejterm_rotated)(centre_model[ibvis], bvis)
                       for ibvis, bvis in enumerate(bvis_graph)]
            vp_list = rsexecute.persist(vp_list)

            dirty_list = \
                calculate_residual_fft_rsexecute_workflow(bvis_graph, original_components,
                                                          future_model_list, vp_list=vp_list,
                                                          context=imaging_context,
                                                          vis_slices=vis_slices,
                                                          do_wstacking=False)
            dirty_list = rsexecute.persist(dirty_list)

        dirty_list = rsexecute.compute(dirty_list, sync=True)
        
        for ipol, pol in enumerate(["I", "Q", "U", "V"]):
            result["onsource_model_{}".format(pol)] = flux[0][ipol]
        for ipol, pol in enumerate(["I", "Q", "U", "V"]):
            polimage = copy_image(dirty_list[0])
            polimage.data = polimage.data[:, ipol, ...][:, numpy.newaxis, ...]
            qa = qa_image(polimage, context="Stokes " + pol)
            result["onsource_{}_{}".format(method, pol)] = max(qa.data['min'], qa.data['max'], key=abs)
        export_image_to_fits(dirty_list[0],
                             "{}/test_voltage_pattern_pol_rsexecute_{}_{}.fits".format(self.dir, name, method))
        
        return result
    
    def check_values(self, result, method, tolerance=1e-3):
        # from pprint import PrettyPrinter
        # pp = PrettyPrinter()
        # pp.pprint(result)
       for ipol, pol in enumerate(["I", "Q", "U", "V"]):
            error = result["onsource_{}_{}".format(method, pol)] - result["onsource_model_{}".format(pol)]
            assert abs(error) < tolerance, "Stokes {}, error too large: {}".format(pol, error)
    
    def test_apply_voltage_pattern_image_unit_dft(self):
        result = self._test(name="unit", method="dft")
        self.check_values(result, method="dft", tolerance=1e-3)
    
    def test_apply_voltage_pattern_image_unit_fft(self):
        result = self._test(name="unit", method="fft")
        self.check_values(result, method="fft", tolerance=3e-3)
    
    def test_apply_voltage_pattern_image_unit_test_vp_dft(self):
        result = self._test(test_vp=True, method="dft", name="unit_test_vp")
        self.check_values(result, method="dft", tolerance=1e-12)
    
    def test_apply_voltage_pattern_image_unit_test_vp_fft(self):
        result = self._test(test_vp=True, method="fft", name="unit_test_vp")
        self.check_values(result, method="fft", tolerance=1e-12)
    
    def test_apply_voltage_pattern_image_nonunit_test_vp_dft(self):
        result = self._test(test_vp=True, method="dft", name="nonunit_test_vp", flux=[1.0, 0.5, -0.2, 0.1])
        self.check_values(result, method="dft", tolerance=1e-12)
    
    def test_apply_voltage_pattern_image_nonunit_test_vp_fft(self):
        result = self._test(test_vp=True, method="fft", name="nonunit_test_vp", flux=[1.0, 0.5, -0.2, 0.1])
        self.check_values(result, method="fft", tolerance=1e-12)


if __name__ == '__main__':
    unittest.main()
