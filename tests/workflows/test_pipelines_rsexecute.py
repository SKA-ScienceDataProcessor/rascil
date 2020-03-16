"""Unit tests for pipelines expressed via dask.delayed


"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.data_model_helpers import export_gaintable_to_hdf5
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.calibration.chain_calibration import create_calibration_controls
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from rascil.processing_components.image.operations import export_image_to_fits, qa_image, smooth_image
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from rascil.processing_components.simulation import simulate_gaintable
from rascil.processing_components.skycomponent.operations import insert_skycomponent
from rascil.processing_components.visibility.coalesce import convert_blockvisibility_to_visibility
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.pipelines.pipeline_rsexecute import ical_list_rsexecute_workflow, \
    continuum_imaging_list_rsexecute_workflow

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))

class TestPipelineGraphs(unittest.TestCase):
    
    def setUp(self):
        rsexecute.set_client(use_dask=True)
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def tearDown(self):
        rsexecute.close()
    
    def actualSetUp(self, add_errors=False, nfreqwin=7, dospectral=True, dopol=False, zerow=True):
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = nfreqwin
        self.vis_list = list()
        self.ntimes = 5
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
        self.blockvis_list = rsexecute.scatter(self.blockvis_list)
        
        self.vis_list = [rsexecute.execute(convert_blockvisibility_to_visibility, nout=1)(bv) for bv in
                         self.blockvis_list]
        self.vis_list = rsexecute.compute(self.vis_list, sync=True)
        self.vis_list = rsexecute.scatter(self.vis_list)
        
        self.model_imagelist = [rsexecute.execute(create_unittest_model, nout=1)
                                (self.vis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005)
                                for i in range(nfreqwin)]
        self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
        self.model_imagelist = rsexecute.scatter(self.model_imagelist)
        
        self.components_list = [rsexecute.execute(create_unittest_components)
                                (self.model_imagelist[freqwin], flux[freqwin, :][numpy.newaxis, :])
                                for freqwin, m in enumerate(self.model_imagelist)]
        self.components_list = rsexecute.compute(self.components_list, sync=True)
        self.components_list = rsexecute.scatter(self.components_list)
        
        self.blockvis_list = [rsexecute.execute(dft_skycomponent_visibility)
                              (self.blockvis_list[freqwin], self.components_list[freqwin])
                              for freqwin, _ in enumerate(self.blockvis_list)]
        self.blockvis_list = rsexecute.compute(self.blockvis_list, sync=True)
        self.vis = self.blockvis_list[0]
        self.blockvis_list = rsexecute.scatter(self.blockvis_list)
        
        self.model_imagelist = [rsexecute.execute(insert_skycomponent, nout=1)
                                (self.model_imagelist[freqwin], self.components_list[freqwin])
                                for freqwin in range(nfreqwin)]
        self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
        model = self.model_imagelist[0]
        self.cmodel = smooth_image(model)
        if self.persist:
            export_image_to_fits(model, '%s/test_pipelines_rsexecute_model.fits' % self.dir)
            export_image_to_fits(self.cmodel, '%s/test_pipelines_rsexecute_cmodel.fits' % self.dir)
        
        if add_errors:
            gt = create_gaintable_from_blockvisibility(self.vis)
            gt = simulate_gaintable(gt, phase_error=0.1, amplitude_error=0.0, smooth_channels=1, leakage=0.0)
            self.blockvis_list = [rsexecute.execute(apply_gaintable, nout=1)
                                  (self.blockvis_list[i], gt)
                                  for i in range(self.freqwin)]
            self.blockvis_list = rsexecute.compute(self.blockvis_list, sync=True)
            self.blockvis_list = rsexecute.scatter(self.blockvis_list)
        
        self.vis_list = [rsexecute.execute(convert_blockvisibility_to_visibility)(bv) for bv in self.blockvis_list]
        self.vis_list = rsexecute.compute(self.vis_list, sync=True)
        self.vis_list = rsexecute.scatter(self.vis_list)
        
        self.model_imagelist = [rsexecute.execute(create_unittest_model, nout=1)
                                (self.vis_list[i], self.image_pol, npixel=self.npixel, cellsize=0.0005)
                                for i in range(nfreqwin)]
        self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
        self.model_imagelist = rsexecute.scatter(self.model_imagelist)
    
    def test_continuum_imaging_pipeline(self):
        self.actualSetUp(add_errors=False, zerow=True)
        continuum_imaging_list = \
            continuum_imaging_list_rsexecute_workflow(self.vis_list,
                                                      model_imagelist=self.model_imagelist,
                                                      context='2d',
                                                      algorithm='mmclean', facets=1,
                                                      scales=[0, 3, 10],
                                                      niter=1000, fractional_threshold=0.1, threshold=0.1,
                                                      nmoment=3,
                                                      nmajor=5, gain=0.1,
                                                      deconvolve_facets=4, deconvolve_overlap=32,
                                                      deconvolve_taper='tukey', psf_support=64,
                                                      restore_facets=4, psfwidth=1.0)
        clean, residual, restored = rsexecute.compute(continuum_imaging_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_clean.fits' %
                                 self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 99.96056316339507) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.4027437530187419) < 1.0e-7, str(qa)
    
    def test_ical_pipeline(self):
        self.actualSetUp(add_errors=False)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'
        
        ical_list = \
            ical_list_rsexecute_workflow(self.vis_list,
                                         model_imagelist=self.model_imagelist,
                                         context='2d',
                                         algorithm='mmclean', facets=1,
                                         scales=[0, 3, 10],
                                         niter=1000, fractional_threshold=0.1, threshold=0.1,
                                         nmoment=3,
                                         nmajor=5, gain=0.1,
                                         deconvolve_facets=4, deconvolve_overlap=32,
                                         deconvolve_taper='tukey', psf_support=64,
                                         restore_facets=4, psfwidth=1.0,
                                         calibration_context='T', controls=controls, do_selfcal=True,
                                         global_solution=False)
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_pipelines_ical_pipeline_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_pipeline_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre], '%s/test_pipelines_ical_pipeline_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[centre]['T'], '%s/test_pipelines_ical_pipeline_rsexecute_gaintable.hdf5' %
                                     self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 99.96329339612933) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.39885052949469246) < 1.0e-7, str(qa)
    
    def test_ical_pipeline_global(self):
        self.actualSetUp(add_errors=False)
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['timeslice'] = 'auto'
        
        ical_list = \
            ical_list_rsexecute_workflow(self.vis_list,
                                         model_imagelist=self.model_imagelist,
                                         context='2d',
                                         algorithm='mmclean', facets=1,
                                         scales=[0, 3, 10],
                                         niter=1000, fractional_threshold=0.1, threshold=0.1,
                                         nmoment=3,
                                         nmajor=5, gain=0.1,
                                         deconvolve_facets=4, deconvolve_overlap=32,
                                         deconvolve_taper='tukey', psf_support=64,
                                         restore_facets=4, psfwidth=1.0,
                                         calibration_context='T', controls=controls, do_selfcal=True,
                                         global_solution=True)
        clean, residual, restored, gt_list = rsexecute.compute(ical_list, sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_ical_global_pipeline_rsexecute_clean.fits' % self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_ical_global_pipeline_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_ical_global_pipeline_rsexecute_restored.fits' % self.dir)
            export_gaintable_to_hdf5(gt_list[0]['T'],
                                     '%s/test_pipelines_ical_global_pipeline_rsexecute_gaintable.hdf5' %
                                     self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 99.96167141746571) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.40137591446730764) < 1.0e-7, str(qa)
    
    def test_continuum_imaging_pipeline_serialclean(self):
        self.actualSetUp(add_errors=False, zerow=True)
        continuum_imaging_list = \
            continuum_imaging_list_rsexecute_workflow(self.vis_list,
                                                      model_imagelist=self.model_imagelist,
                                                      context='2d',
                                                      algorithm='mmclean',
                                                      facets=1,
                                                      scales=[0, 3, 10],
                                                      niter=1000,
                                                      fractional_threshold=0.1,
                                                      threshold=0.1,
                                                      nmoment=3,
                                                      nmajor=5, gain=0.1,
                                                      deconvolve_facets=4,
                                                      deconvolve_overlap=32,
                                                      deconvolve_taper='tukey',
                                                      psf_support=64,
                                                      use_serial_clean=True,
                                                      restore_facets=4,
                                                      psfwidth=1.0)
        clean, residual, restored = rsexecute.compute(continuum_imaging_list,
                                                      sync=True)
        centre = len(clean) // 2
        if self.persist:
            export_image_to_fits(clean[centre],
                                 '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_clean.fits' %
                                 self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_pipelines_continuum_imaging_pipeline_rsexecute_restored.fits' % self.dir)
        
        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 99.96050848972253) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.40249368636733457) < 1.0e-7, str(qa)


if __name__ == '__main__':
    unittest.main()
