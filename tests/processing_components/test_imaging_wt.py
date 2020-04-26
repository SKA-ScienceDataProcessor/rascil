""" Unit tests for imaging using wtowers

"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.griddata import create_awterm_convolutionfunction
from rascil.processing_components.image.operations import export_image_to_fits, smooth_image
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from rascil.processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent, apply_beam_to_skycomponent
from rascil.processing_components.visibility import copy_visibility

try:
    import wtowers
    
    run_wt_tests = True
#            except ModuleNotFoundError:
except ImportError:
    run_wt_tests = False

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingWT(unittest.TestCase):
    def setUp(self):
        
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", True)
        
        self.verbosity = 0
    
    def actualSetUp(self, freqwin=1, block=True, dospectral=True,
                    image_pol=PolarisationFrame('stokesI'), zerow=False, make_pb=None):
        
        self.npixel = 512
        self.low = create_named_configuration('LOWBD2', rmax=750.0)
        self.freqwin = freqwin
        self.vis = list()
        self.ntimes = 5
        self.times = numpy.linspace(-3.0, +3.0, self.ntimes) * numpy.pi / 12.0
        
        if freqwin > 1:
            self.frequency = numpy.linspace(0.8e8, 1.2e8, self.freqwin)
            self.channelwidth = numpy.array(freqwin * [self.frequency[1] - self.frequency[0]])
        else:
            self.frequency = numpy.array([1e8])
            self.channelwidth = numpy.array([1e6])
        
        if image_pol == PolarisationFrame('stokesIQUV'):
            self.vis_pol = PolarisationFrame('linear')
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
        elif image_pol == PolarisationFrame('stokesIQ'):
            self.vis_pol = PolarisationFrame('linearnp')
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0])
        elif image_pol == PolarisationFrame('stokesIV'):
            self.vis_pol = PolarisationFrame('circularnp')
            self.image_pol = image_pol
            f = numpy.array([100.0, 20.0])
        else:
            self.vis_pol = PolarisationFrame('stokesI')
            self.image_pol = PolarisationFrame('stokesI')
            f = numpy.array([100.0])
        
        if dospectral:
            flux = numpy.array([f * numpy.power(freq / 1e8, -0.7) for freq in self.frequency])
        else:
            flux = numpy.array([f])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = ingest_unittest_visibility(self.low,
                                              self.frequency,
                                              self.channelwidth,
                                              self.times,
                                              self.vis_pol,
                                              self.phasecentre,
                                              block=block,
                                              zerow=zerow)
        
        self.model = create_unittest_model(self.vis, self.image_pol, npixel=self.npixel, nchan=freqwin)
        
        self.components = create_unittest_components(self.model, flux)
        
        self.model = insert_skycomponent(self.model, self.components)
        
        # We need to apply primary beam to the components to get the correct predicted
        # visibility to compare with the value predicted by degridding
        if make_pb is not None:
            pb = make_pb(self.model)
            self.components = apply_beam_to_skycomponent(self.components, pb)
        
        self.vis = dft_skycomponent_visibility(self.vis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        
        self.cmodel = smooth_image(self.model)
        if self.persist:
            export_image_to_fits(self.model, '%s/test_imaging_wt_model.fits' % self.dir)
            export_image_to_fits(self.cmodel, '%s/test_imaging_wt_cmodel.fits' % self.dir)
        
        self.gcfcf = create_awterm_convolutionfunction(self.model, make_pb=None, nw=101, wstep=6, oversampling=8,
                                                       support=32, use_aaf=False, maxsupport=512, wtowers=True)
    
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=0.1, flux_difference=5.0):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        
        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(comp.direction, self.components)
            assert separation / cellsize < positionthreshold, "Component differs in position %.3f pixels" % \
                                                              separation / cellsize
            assert numpy.abs(ocomp.flux[0, 0] - comp.flux[0, 0]) < flux_difference, \
                "Component flux {:.3f} differs from original {:.3f}".format(comp.flux[0, 0], ocomp.flux[0, 0])
    
    def _predict_base(self, fluxthreshold=1.0, name='predict_wt', **kwargs):
        
        from rascil.processing_components.imaging.wt import predict_wt, invert_wt
        original_vis = copy_visibility(self.vis)
        vis = predict_wt(self.vis, self.model, verbosity=self.verbosity, grid_reference=0.5, **kwargs)
        vis.data['vis'] = vis.data['vis'] - original_vis.data['vis']
        dirty = invert_wt(vis, self.model, dopsf=False, normalize=True, grid_reference=0.5, **kwargs)
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_wt_%s_residual.fits' %
                                              (self.dir, name))
        
        maxabs = numpy.max(numpy.abs(dirty[0].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (maxabs, fluxthreshold)
    
    def _invert_base(self, fluxthreshold=1.0, positionthreshold=1.0, check_components=True,
                     name='invert_wt', gcfcf=None, flux_difference=5.0, **kwargs):
        
        from rascil.processing_components.imaging.wt import invert_wt
        dirty = invert_wt(self.vis, self.model, normalize=True, verbosity=self.verbosity,
                          gcfcf=gcfcf, **kwargs)
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_wt_%s_dirty.fits' %
                                              (self.dir, name))
        
        assert numpy.max(numpy.abs(dirty[0].data)), "Image is empty"
        
        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold, flux_difference=flux_difference)
    
    @unittest.skipUnless(run_wt_tests, "requires the py-wtowers module")
    def test_predict_wt(self):
        self.actualSetUp()
        self._predict_base(name='predict_wt', gcfcf=self.gcfcf, NpixFF=512, fluxthreshold=1.3)
    
    @unittest.skipUnless(run_wt_tests, "requires the py-wtowers module")
    def test_invert_wt(self):
        self.actualSetUp()
        self._invert_base(name='invert_wt', positionthreshold=0.1, check_components=True,
                          gcfcf=self.gcfcf, NpixFF=512, fluxthreshold=1.0)
    @unittest.skipUnless(run_wt_tests, "requires the py-wtowers module")
    def test_invert_wt_wtowers(self):
        self.actualSetUp()
        self._invert_base(name='invert_wt_wtowers', positionthreshold=0.1, check_components=True,
                          gcfcf=self.gcfcf, NpixFF=512, fluxthreshold=1.0,
                          subgrid_size=100, margin=36, winc=6)

    @unittest.skipUnless(run_wt_tests, "requires the py-wtowers module")
    def test_predict_wt_wtowers(self):
        self.actualSetUp()
        self._predict_base(name='predict_wt_wtowers', gcfcf=self.gcfcf, NpixFF=512, fluxthreshold=1.3, 
                          subgrid_size=100, margin=36, winc=6)



if __name__ == '__main__':
    unittest.main()
