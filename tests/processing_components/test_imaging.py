""" Unit tests for pipelines expressed via dask.delayed


"""
import functools
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.griddata.kernels import create_awterm_convolutionfunction
from rascil.processing_components.image.operations import export_image_to_fits, smooth_image, qa_image
from rascil.processing_components.imaging.base import predict_2d, invert_2d
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility
from rascil.processing_components.imaging.primary_beams import create_pb_generic
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from rascil.processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent
from rascil.processing_components.visibility import copy_visibility

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImaging2D(unittest.TestCase):
    def setUp(self):
        
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", True)
    
    def actualSetUp(self, freqwin=1, block=False, dospectral=True,
                    image_pol=PolarisationFrame('stokesI'), zerow=False):
        
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
        
        self.vis = dft_skycomponent_visibility(self.vis, self.components)
        
        # Calculate the model convolved with a Gaussian.
        
        self.cmodel = smooth_image(self.model)
        if self.persist: export_image_to_fits(self.model, '%s/test_imaging_model.fits' % self.dir)
        if self.persist: export_image_to_fits(self.cmodel, '%s/test_imaging_cmodel.fits' % self.dir)
    
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=0.1):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))
        cellsize = abs(dirty.wcs.wcs.cdelt[0])
        
        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(comp.direction, self.components)
            assert separation / cellsize < positionthreshold, "Component differs in position %.3f pixels" % \
                                                              separation / cellsize
    
    def _predict_base(self, fluxthreshold=1.0, name='predict_2d', gcfcf=None, **kwargs):
        
        vis = predict_2d(self.vis, self.model, gcfcf=gcfcf, **kwargs)
        vis.data['vis'] = self.vis.data['vis'] - vis.data['vis']
        dirty = invert_2d(vis, self.model, dopsf=False, normalize=True, gcfcf=gcfcf)
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_%s_residual.fits' %
                                              (self.dir, name))
        # assert numpy.max(numpy.abs(dirty[0].data)), "Residual image is empty"
        
        maxabs = numpy.max(numpy.abs(dirty[0].data))
        assert maxabs < fluxthreshold, "Error %.3f greater than fluxthreshold %.3f " % (maxabs, fluxthreshold)
    
    def _invert_base(self, fluxthreshold=1.0, positionthreshold=1.0, check_components=True,
                     name='invert_2d', gcfcf=None, **kwargs):
        
        dirty = invert_2d(self.vis, self.model, dopsf=False, normalize=True, gcfcf=gcfcf,
                          **kwargs)
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_%s_dirty.fits' %
                                              (self.dir, name))
        
        assert numpy.max(numpy.abs(dirty[0].data)), "Image is empty"
        
        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)
    
    def test_predict_2d(self):
        self.actualSetUp(zerow=True)
        self._predict_base(name='predict_2d')
    
    def test_predict_2d_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self._predict_base(name='predict_2d_IQUV')
    
    def test_predict_2d_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        self._predict_base(name='predict_2d_IQ')
    
    def test_predict_2d_IV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIV"))
        self._predict_base(name='predict_2d_IV')
    
    def test_invert_2d(self):
        self.actualSetUp(zerow=True)
        self._invert_base(name='invert_2d', positionthreshold=2.0, check_components=True)
    
    def test_invert_2d_IQUV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQUV"))
        self._invert_base(name='invert_2d_IQUV', positionthreshold=2.0, check_components=True)
    
    def test_invert_2d_spec_IQUV(self):
        self.actualSetUp(zerow=True, freqwin=5, image_pol=PolarisationFrame("stokesIQUV"))
        self._invert_base(name='invert_2d_IQUV', positionthreshold=2.0, check_components=True)
    
    def test_invert_2d_IQ(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIQ"))
        self._invert_base(name='invert_2d_IQ', positionthreshold=2.0, check_components=True)
    
    def test_invert_2d_IV(self):
        self.actualSetUp(zerow=True, image_pol=PolarisationFrame("stokesIV"))
        self._invert_base(name='invert_2d_IV', positionthreshold=2.0, check_components=True)
    
    def test_predict_2d_block(self):
        self.actualSetUp(zerow=True, block=True)
        self._predict_base(name='predict_2d_block')
    
    def test_invert_2d_block(self):
        self.actualSetUp(zerow=True, block=True)
        self._invert_base(name='invert_2d_block', positionthreshold=2.0, check_components=True)
    
    def test_predict_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._predict_base(fluxthreshold=35.0, name='predict_awterm', gcfcf=gcfcf)

    def test_predict_awterm_spec(self):
        self.actualSetUp(zerow=False, freqwin=5)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._predict_base(fluxthreshold=35.0, name='predict_awterm_spec', gcfcf=gcfcf)

    def test_predict_awterm_spec_IQUV(self):
        self.actualSetUp(zerow=False, freqwin=5, image_pol=PolarisationFrame("stokesIQUV"))
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._predict_base(fluxthreshold=35.0, name='predict_awterm_spec_IQUV', gcfcf=gcfcf)

    def test_invert_awterm(self):
        self.actualSetUp(zerow=False)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._invert_base(name='invert_awterm', positionthreshold=35.0, check_components=False, gcfcf=gcfcf)

    def test_invert_awterm_spec(self):
        self.actualSetUp(zerow=False, freqwin=5)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._invert_base(name='invert_awterm_spec', positionthreshold=35.0, check_components=False, gcfcf=gcfcf)

    def test_invert_awterm_spec_IQUV(self):
        self.actualSetUp(zerow=False, freqwin=5, image_pol=PolarisationFrame("stokesIQUV"))
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._invert_base(name='invert_awterm_spec_IQUV', positionthreshold=35.0, check_components=False, gcfcf=gcfcf)

    def test_predict_awterm_block(self):
        self.actualSetUp(zerow=False, block=True)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._predict_base(fluxthreshold=35.0, name='predict_awterm_block', gcfcf=gcfcf)

    def test_invert_awterm_block(self):
        self.actualSetUp(zerow=False, block=True)
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        gcfcf = create_awterm_convolutionfunction(self.model, make_pb=make_pb, nw=100, wstep=8.0,
                                                  oversampling=4, support=100, use_aaf=True)
        self._invert_base(name='invert_awterm_block', positionthreshold=35.0, check_components=False, gcfcf=gcfcf)
    
    def test_predict_wterm(self):
        self.actualSetUp(zerow=False)
        gcfcf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0,
                                                  oversampling=8, support=100, use_aaf=True)
        self._predict_base(fluxthreshold=5.0, name='predict_wterm', gcfcf=gcfcf)
    
    def test_invert_wterm(self):
        self.actualSetUp(zerow=False)
        gcfcf = create_awterm_convolutionfunction(self.model, nw=100, wstep=8.0,
                                                  oversampling=8, support=100, use_aaf=True)
        self._invert_base(name='invert_wterm', positionthreshold=35.0, check_components=False, gcfcf=gcfcf)

    def test_invert_psf(self):
        self.actualSetUp(zerow=False)
        psf = invert_2d(self.vis, self.model, dopsf=True)

        if self.persist: export_image_to_fits(psf[0], '%s/test_imaging_2d_psf.fits' % (self.dir))
        
        assert numpy.max(numpy.abs(psf[0].data)), "Image is empty"


if __name__ == '__main__':
    unittest.main()
