""" Unit tests for pipelines expressed via dask.delayed


"""
import os
import logging
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.simulation import create_named_configuration
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import invert_list_rsexecute_workflow, \
    deconvolve_list_rsexecute_workflow, \
    residual_list_rsexecute_workflow, restore_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.processing_components.image.operations import export_image_to_fits, smooth_image
from rascil.processing_components.imaging.base import predict_skycomponent_visibility
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components, insert_unittest_errors
from rascil.processing_components.skycomponent.operations import insert_skycomponent

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingDeconvolveGraph(unittest.TestCase):
    
    def setUp(self):
        rsexecute.set_client(verbose=False, memory_limit=4 * 1024 * 1024 * 1024, n_workers=4, dashboard_address=None)
        
        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def tearDown(self):
        rsexecute.close()

    def actualSetUp(self, add_errors=False, freqwin=7, block=False, dospectral=True, dopol=False,
                    zerow=True):
        
        self.npixel = 256
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis_list = list()
        self.ntimes = 5
        cellsize = 0.001
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
        
        if freqwin > 1:
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
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
        self.vis_list = [rsexecute.execute(ingest_unittest_visibility)(self.low,
                                                                        [self.frequency[freqwin]],
                                                                        [self.channelwidth[freqwin]],
                                                                        self.times,
                                                                        self.vis_pol,
                                                                        self.phasecentre, block=block,
                                                                        zerow=zerow)
                         for freqwin, _ in enumerate(self.frequency)]
        
        self.model_imagelist = [rsexecute.execute(create_unittest_model, nout=freqwin)(self.vis_list[freqwin],
                                                                                        self.image_pol,
                                                                                        cellsize=cellsize,
                                                                                        npixel=self.npixel)
                                for freqwin, _ in enumerate(self.frequency)]
        
        self.componentlist = [rsexecute.execute(create_unittest_components)(self.model_imagelist[freqwin],
                                                                             flux[freqwin, :][numpy.newaxis, :])
                              for freqwin, _ in enumerate(self.frequency)]
        
        self.model_imagelist = [rsexecute.execute(insert_skycomponent, nout=1)(self.model_imagelist[freqwin],
                                                                                self.componentlist[freqwin])
                                for freqwin, _ in enumerate(self.frequency)]
        
        self.vis_list = [rsexecute.execute(predict_skycomponent_visibility)(self.vis_list[freqwin],
                                                                             self.componentlist[freqwin])
                         for freqwin, _ in enumerate(self.frequency)]
        
        # Calculate the model convolved with a Gaussian.
        
        self.model_imagelist = rsexecute.compute(self.model_imagelist, sync=True)
        model = self.model_imagelist[0]
        
        self.cmodel = smooth_image(model)
        if self.persist: export_image_to_fits(model, '%s/test_imaging_rsexecute_deconvolved_model.fits' % self.dir)
        if self.persist: export_image_to_fits(self.cmodel, '%s/test_imaging_rsexecute_deconvolved_cmodel.fits' % self.dir)
        
        if add_errors and block:
            self.vis_list = [rsexecute.execute(insert_unittest_errors)(self.vis_list[i])
                             for i, _ in enumerate(self.frequency)]
        
        #        self.vis_list = rsexecute.compute(self.vis_list, sync=True)
        self.vis_list = rsexecute.persist(self.vis_list)
        self.model_imagelist = rsexecute.scatter(self.model_imagelist)
    
    def test_time_setup(self):
        self.actualSetUp()
    
    def test_deconvolve_spectral(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist, context='2d',
                                                          dopsf=False, normalize=True)
        psf_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist,
                                                        context='2d',
                                                        dopsf=True, normalize=True)
        dirty_imagelist = rsexecute.persist(dirty_imagelist)
        psf_imagelist = rsexecute.persist(psf_imagelist)
        deconvolved = deconvolve_list_rsexecute_workflow(dirty_imagelist, psf_imagelist, self.model_imagelist,
                                                          niter=1000,
                                                          fractional_threshold=0.1, scales=[0, 3, 10],
                                                          threshold=0.1, gain=0.7)
        deconvolved = rsexecute.persist(deconvolved)
        deconvolved = rsexecute.compute(deconvolved, sync=True)
        
        if self.persist: export_image_to_fits(deconvolved[0], '%s/test_imaging_%s_deconvolve_spectral.fits' %
                             (self.dir, rsexecute.type()))
    
    def test_deconvolve_and_restore_cube_mmclean(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist, context='2d',
                                                          dopsf=False, normalize=True)
        psf_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist, context='2d',
                                                        dopsf=True, normalize=True)
        dirty_imagelist = rsexecute.persist(dirty_imagelist)
        psf_imagelist = rsexecute.persist(psf_imagelist)
        dec_imagelist = deconvolve_list_rsexecute_workflow(dirty_imagelist, psf_imagelist, self.model_imagelist,
                                                            niter=1000,
                                                            fractional_threshold=0.01, scales=[0, 3, 10],
                                                            algorithm='mmclean', nmoment=3, nchan=self.freqwin,
                                                            threshold=0.1, gain=0.7)
        dec_imagelist = rsexecute.persist(dec_imagelist)
        residual_imagelist = residual_list_rsexecute_workflow(self.vis_list, model_imagelist=dec_imagelist,
                                                               context='2d')
        residual_imagelist = rsexecute.persist(residual_imagelist)
        restored_list = restore_list_rsexecute_workflow(model_imagelist=dec_imagelist, psf_imagelist=psf_imagelist,
                                                         residual_imagelist=residual_imagelist,
                                                         empty=self.model_imagelist)
        
        restored = rsexecute.compute(restored_list, sync=True)[0]
        
        if self.persist: export_image_to_fits(restored, '%s/test_imaging_%s_mmclean_restored.fits' % (self.dir, rsexecute.type()))
    
    def test_deconvolve_and_restore_cube_mmclean_serialclean(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist, context='2d',
                                                          dopsf=False, normalize=True)
        psf_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist, context='2d',
                                                        dopsf=True, normalize=True)
        dirty_imagelist = rsexecute.persist(dirty_imagelist)
        psf_imagelist = rsexecute.persist(psf_imagelist)
        dec_imagelist = deconvolve_list_rsexecute_workflow(dirty_imagelist, psf_imagelist,
                                                            self.model_imagelist,
                                                            niter=1000, use_serial_clean=True,
                                                            fractional_threshold=0.01, scales=[0, 3, 10],
                                                            algorithm='mmclean', nmoment=3, nchan=self.freqwin,
                                                            threshold=0.1, gain=0.7)
        dec_imagelist = rsexecute.persist(dec_imagelist)
        residual_imagelist = residual_list_rsexecute_workflow(self.vis_list, model_imagelist=dec_imagelist,
                                                               context='2d')
        residual_imagelist = rsexecute.persist(residual_imagelist)
        restored_list = restore_list_rsexecute_workflow(model_imagelist=dec_imagelist, psf_imagelist=psf_imagelist,
                                                         residual_imagelist=residual_imagelist,
                                                         empty=self.model_imagelist)
        
        restored = rsexecute.compute(restored_list, sync=True)[0]
        
        if self.persist: export_image_to_fits(restored, '%s/test_imaging_%s_mmclean_restored.fits' % (self.dir, rsexecute.type()))
    
    def test_deconvolve_and_restore_cube_mmclean_facets(self):
        self.actualSetUp(add_errors=True)
        dirty_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist, context='2d',
                                                          dopsf=False, normalize=True)
        psf_imagelist = invert_list_rsexecute_workflow(self.vis_list, self.model_imagelist, context='2d',
                                                        dopsf=True, normalize=True)
        dirty_imagelist = rsexecute.persist(dirty_imagelist)
        psf_imagelist = rsexecute.persist(psf_imagelist)
        dec_imagelist = deconvolve_list_rsexecute_workflow(dirty_imagelist, psf_imagelist, self.model_imagelist,
                                                            niter=1000,
                                                            fractional_threshold=0.1, scales=[0, 3, 10],
                                                            algorithm='mmclean', nmoment=3, nchan=self.freqwin,
                                                            threshold=0.01, gain=0.7, deconvolve_facets=8,
                                                            deconvolve_overlap=8, deconvolve_taper='tukey')
        dec_imagelist = rsexecute.persist(dec_imagelist)
        residual_imagelist = residual_list_rsexecute_workflow(self.vis_list, model_imagelist=dec_imagelist,
                                                               context='2d')
        residual_imagelist = rsexecute.persist(residual_imagelist)
        restored_list = restore_list_rsexecute_workflow(model_imagelist=dec_imagelist, psf_imagelist=psf_imagelist,
                                                         residual_imagelist=residual_imagelist,
                                                         empty=self.model_imagelist)
        
        restored = rsexecute.compute(restored_list, sync=True)[0]
        
        if self.persist: export_image_to_fits(restored, '%s/test_imaging_%s_overlap_mmclean_restored.fits'
                             % (self.dir, rsexecute.type()))


if __name__ == '__main__':
    unittest.main()
