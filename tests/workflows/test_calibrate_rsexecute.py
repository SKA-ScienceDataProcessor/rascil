"""Unit tests for pipelines expressed via dask.delayed


"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.calibration.chain_calibration import create_calibration_controls
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.visibility.base import copy_visibility
from rascil.workflows.rsexecute.calibration.calibration_rsexecute import calibrate_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))

log.setLevel(logging.WARNING)


class TestCalibrateGraphs(unittest.TestCase):
    
    def setUp(self):
        rsexecute.set_client(verbose=False, memory_limit=4 * 1024 * 1024 * 1024, n_workers=4,
                            dashboard_address=None)
        
        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def tearDown(self):
        rsexecute.close()
    
    def actualSetUp(self, nfreqwin=3, dospectral=True, dopol=False,
                    amp_errors=None, phase_errors=None, zerow=True):
        
        if amp_errors is None:
            amp_errors = {'T': 0.0, 'G': 0.1}
        if phase_errors is None:
            phase_errors = {'T': 1.0, 'G': 0.0}
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = nfreqwin
        self.vis_list = list()
        self.ntimes = 1
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if self.freqwin > 1:
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.channelwidth = numpy.array([1e6])
        
        if dopol:
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = PolarisationFrame('stokesIQUV')
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.blockvis_list = \
            [rsexecute.execute(ingest_unittest_visibility, nout=1)(self.low,
                                                                   [self.frequency[i]],
                                                                   [self.channelwidth[i]],
                                                                   self.times,
                                                                   self.vis_pol,
                                                                   self.phasecentre, block=True,
                                                                   zerow=zerow)
             for i in range(nfreqwin)]
        self.blockvis_list = rsexecute.compute(self.blockvis_list, sync=True)
        
        for v in self.blockvis_list:
            v.data['vis'][...] = 1.0 + 0.0j
        
        self.error_blockvis_list = [rsexecute.execute(copy_visibility(v)) for v in self.blockvis_list]
        gt = rsexecute.execute(create_gaintable_from_blockvisibility)(self.blockvis_list[0])
        gt = rsexecute.execute(simulate_gaintable)\
            (gt, phase_error=0.1, amplitude_error=0.0, smooth_channels=1, leakage=0.0, seed=180555)
        self.error_blockvis_list = [rsexecute.execute(apply_gaintable)(self.error_blockvis_list[i], gt)
                                    for i in range(self.freqwin)]
        
        self.error_blockvis_list = rsexecute.compute(self.error_blockvis_list, sync=True)
        
        assert numpy.max(numpy.abs(self.error_blockvis_list[0].vis - self.blockvis_list[0].vis)) > 0.0
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_calibrate_rsexecute(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_rsexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                              calibration_context='T', controls=controls,
                                              do_selfcal=True,
                                              global_solution=False)
        calibrate_list = rsexecute.compute(calibrate_list, sync=True)
        
        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        err = numpy.max(numpy.abs(calibrate_list[0][0].flagged_vis - self.blockvis_list[0].flagged_vis))
        assert err < 2e-6, err
    
    def test_calibrate_rsexecute_repeat(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_rsexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                              calibration_context='T', controls=controls, do_selfcal=True,
                                              global_solution=False)
        calibrate_list = rsexecute.compute(calibrate_list, sync=True)
        
        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        err = numpy.max(numpy.abs(calibrate_list[0][0].flagged_vis - self.blockvis_list[0].flagged_vis))
        assert err < 2e-6, err
        
        calibrate_list = \
            calibrate_list_rsexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                              gt_list=calibrate_list[1],
                                              calibration_context='T', controls=controls, do_selfcal=True,
                                              global_solution=False)
        calibrate_list = rsexecute.compute(calibrate_list, sync=True)
        
        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        err = numpy.max(numpy.abs(calibrate_list[0][0].flagged_vis - self.blockvis_list[0].flagged_vis))
        assert err < 2e-6, err
    
    def test_calibrate_rsexecute_empty(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        for v in self.blockvis_list:
            v.data['vis'][...] = 0.0 + 0.0j
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_rsexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                              calibration_context='T', controls=controls, do_selfcal=True,
                                              global_solution=False)
        calibrate_list = rsexecute.compute(calibrate_list, sync=True)
        assert len(calibrate_list[1][0]) > 0
    
    def test_calibrate_rsexecute_global(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_rsexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                              calibration_context='T', controls=controls, do_selfcal=True,
                                              global_solution=True)
        
        calibrate_list = rsexecute.compute(calibrate_list, sync=True)
        
        assert len(calibrate_list) == 2
        assert numpy.max(calibrate_list[1][0]['T'].residual) < 7e-6, numpy.max(calibrate_list[1][0]['T'].residual)
        err = numpy.max(numpy.abs(calibrate_list[0][0].flagged_vis - self.blockvis_list[0].flagged_vis))
        assert err < 2e-6, err
    
    def test_calibrate_rsexecute_global_empty(self):
        amp_errors = {'T': 0.0, 'G': 0.0}
        phase_errors = {'T': 1.0, 'G': 0.0}
        self.actualSetUp(amp_errors=amp_errors, phase_errors=phase_errors)
        
        for v in self.blockvis_list:
            v.data['vis'][...] = 0.0 + 0.0j
        
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 0
        controls['T']['timeslice'] = 'auto'
        
        calibrate_list = \
            calibrate_list_rsexecute_workflow(self.error_blockvis_list, self.blockvis_list,
                                              calibration_context='T', controls=controls, do_selfcal=True,
                                              global_solution=True)
        
        calibrate_list = rsexecute.compute(calibrate_list, sync=True)
        assert len(calibrate_list[1][0]) > 0


if __name__ == '__main__':
    unittest.main()
