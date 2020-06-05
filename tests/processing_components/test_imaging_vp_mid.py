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
from rascil.processing_components import convert_azelvp_to_radec
from rascil.processing_components.griddata.kernels import create_vpterm_convolutionfunction
from rascil.processing_components.griddata import convert_convolutionfunction_to_image
from rascil.processing_components.image.operations import export_image_to_fits, smooth_image, fft_image, copy_image, \
    qa_image
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility
from rascil.processing_components.imaging.imaging_vp import predict_vp, invert_vp
from rascil.processing_components.imaging.weighting import weight_blockvisibility, taper_visibility_gaussian
from rascil.processing_components.imaging.primary_beams import create_vp, create_pb
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from rascil.processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent, apply_beam_to_skycomponent
from rascil.processing_components.visibility import copy_visibility

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))


class TestImagingVP(unittest.TestCase):
    def setUp(self):
        
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        self.persist = os.getenv("RASCIL_PERSIST", True)
    
    def actualSetUp(self, zerow=True, block=False, image_pol=PolarisationFrame("stokesIQUV"), npixel=None, rmax=2400.0,
                    scale=0.25, cellsize=None):
        self.doplot = False
        self.npixel = npixel
        self.low = create_named_configuration('MIDR5', rmax=rmax)
        self.freqwin = 1
        self.vis_list = list()
        self.ntimes = 11
        self.times = numpy.linspace(-4.0, +4.0, self.ntimes) * numpy.pi / 12.0
        
        if self.freqwin == 1:
            self.frequency = numpy.array([1e9])
            self.channelwidth = numpy.array([4e8])
        else:
            self.frequency = numpy.linspace(0.8e9, 1.2e9, self.freqwin)
            self.channelwidth = numpy.array(self.freqwin * [self.frequency[1] - self.frequency[0]])
        
        self.image_pol = image_pol
        if image_pol == PolarisationFrame("stokesI"):
            self.vis_pol = PolarisationFrame("stokesI")
            f = numpy.array([100.0])
        elif image_pol == PolarisationFrame("stokesIQUV"):
            self.vis_pol = PolarisationFrame("linear")
            f = numpy.array([100.0, 20.0, -10.0, 1.0])
            f = numpy.array([100.0, 0.0, 0.0, 0.0])
        elif image_pol == PolarisationFrame("stokesIQ"):
            self.vis_pol = PolarisationFrame("linearnp")
            f = numpy.array([100.0, 20.0])
        elif image_pol == PolarisationFrame("stokesIV"):
            self.vis_pol = PolarisationFrame("circularnp")
            f = numpy.array([100.0, 20.0])
        else:
            raise ValueError("Polarisation {} not supported".format(image_pol))
        
        flux = numpy.array([f * numpy.power(freq / 1e9, -0.7) for freq in self.frequency])
        
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = ingest_unittest_visibility(self.low,
                                              self.frequency,
                                              self.channelwidth,
                                              self.times,
                                              self.vis_pol,
                                              self.phasecentre,
                                              block=block,
                                              zerow=zerow)
        
        from rascil.processing_components import advise_wide_field
        advice=advise_wide_field(self.vis, guard_band_image=4)
        if npixel is None:
            self.npixel = advice['npixels23']
            print("Npixel = ", self.npixel)
        if cellsize is None:
            self.cellsize = advice['cellsize']

        self.model = create_unittest_model(self.vis, self.image_pol, cellsize=self.cellsize,
                                           npixel=self.npixel, nchan=self.freqwin)
        
        self.vis = weight_blockvisibility(self.vis, self.model)
        self.vis = taper_visibility_gaussian(self.vis, 3.0 * self.cellsize)
        self.components = create_unittest_components(self.model, flux, applypb=False,
                                                     scale=scale, single=False, symmetric=False)

        def find_vp_renormalised(model, telescope="MID_FEKO_B2"):
            vp = create_vp(telescope=telescope)
            g = numpy.zeros([4])
            g[0] = numpy.max(numpy.abs(vp.data[:, 0, ...]))
            g[3] = numpy.max(numpy.abs(vp.data[:, 3, ...]))
            g[1] = g[2] = numpy.sqrt(g[0] * g[3])
            for pol in range(4):
                vp.data[:, pol, ...] /= g[pol]
            vp = convert_azelvp_to_radec(vp, model, pa=0.0)
            return vp

        self.diameter = 35.0
        self.pb = create_pb(self.model, 'MID_FEKO_B2')
        self.pb = convert_azelvp_to_radec(self.pb, self.model, pa=0.0)
        self.pb_components = apply_beam_to_skycomponent(self.components, self.pb)
        self.pb_model = copy_image(self.model)
        self.pb_model.data[...] = 0.0
        self.pb_model = insert_skycomponent(self.pb_model, self.pb_components)
        self.model = insert_skycomponent(self.model, self.components)
        self.vis = dft_skycomponent_visibility(self.vis, self.components)
        make_vp = functools.partial(create_vp, telescope='MID_FEKO_B2')
        self.vp = make_vp(self.model)
        self.vp = convert_azelvp_to_radec(self.vp, self.model, pa=0.0)
        export_image_to_fits(self.vp, "%s/test_imaging_vp_mid_vp.fits" % self.dir)

        self.gcf, self.cf = create_vpterm_convolutionfunction(self.model,
                                                              make_vp=find_vp_renormalised,
                                                              oversampling=33,
                                                              support=16,
                                                              use_aaf=False)
        cf_image = convert_convolutionfunction_to_image(self.cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_imaging_vp_mid_cf.fits" % self.dir)
        
        self.pb_cmodel = smooth_image(self.pb_model)
        if self.persist:
            export_image_to_fits(self.model, '%s/test_imaging_model.fits' % self.dir)
            export_image_to_fits(self.pb_model, '%s/test_imaging_pb_model.fits' % self.dir)
            export_image_to_fits(self.pb_cmodel, '%s/test_imaging_pb_cmodel.fits' % self.dir)
        self.peak = numpy.unravel_index(numpy.argmax(numpy.abs(self.pb_cmodel.data)), self.pb_cmodel.shape)
        
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=0.1):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=10 * fluxthreshold, npixels=5)
        
        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(comp.direction, self.pb_components)
            assert separation / self.cellsize < positionthreshold, \
                "Component differs in position {} pixels".format(separation / self.cellsize)
            err = numpy.max(numpy.abs(comp.flux - ocomp.flux))
            #assert err < fluxthreshold, err
            print(comp.flux, ocomp.flux, comp.flux-ocomp.flux, err)
        assert len(comps) == len(self.components), "Different number of components found: original %d, recovered %d" % \
                                                   (len(self.components), len(comps))

    def _predict_base(self, fluxthreshold=0.05, name='predict_vp', **kwargs):
        self.vis.data['vis'][...] = 0.0
        self.vis = dft_skycomponent_visibility(self.vis, self.pb_components)

        original_vis = copy_visibility(self.vis)
        self.vis.data['vis'][...] = 0.0
        vis = predict_vp(self.vis, self.model, vp=self.vp, cf=self.cf, **kwargs)
        vis.data['vis'] = original_vis.data['vis'] - vis.data['vis']
        dirty = invert_vp(vis, self.model, dopsf=False, normalize=True, vp=self.vp, cf=self.cf)
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_vp_mid_%s_residual.fits' %
                                              (self.dir, name))
        for pol in range(dirty[0].npol):
            assert numpy.max(numpy.abs(dirty[0].data[:, pol])), "Residual image pol {} is empty".format(pol)
        
        for pol in range(4):
            maxabs = numpy.max(numpy.abs(dirty[0].data[:, pol]))
            pol_name = dirty[0].polarisation_frame.names[pol]
            assert maxabs < fluxthreshold, \
                "Error %.3f in pol %s greater than fluxthreshold %.3f " % (maxabs, pol_name, fluxthreshold)
        self.dirty = dirty
    
    def _invert_base(self, fluxthreshold=0.6, positionthreshold=1.0, check_components=True,
                     name='invert_vp', **kwargs):
        
        self.model.data[...] = 0.0
        dirty = invert_vp(self.vis, self.model, normalize=True, vp=self.vp, cf=self.cf, **kwargs)
        self.dirty = dirty
        
        if self.persist:
            export_image_to_fits(dirty[0], '%s/test_imaging_vp_mid_%s_dirty.fits' % (self.dir, name))
            dirty_fft = fft_image(dirty[0])
            export_image_to_fits(dirty_fft, '%s/test_imaging_vp_mid_%s_dirty_fft.fits' % (self.dir, name))

        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)
    
    def test_predict_vp(self):
        self.actualSetUp(zerow=True, block=True)
        self._predict_base(name='predict_vp', fluxthreshold=0.2)
    
    def test_invert_vp(self):
        self.actualSetUp(zerow=True, block=True)
        self._invert_base(name='invert_vp', positionthreshold=2.0, check_components=True, fluxthreshold=1.0)

    def test_invert_vp_psf(self):
        self.actualSetUp(zerow=True, block=True)
        self._invert_base(name='invert_vp_psf', positionthreshold=2.0, check_components=False, fluxthreshold=1.0,
                          dopsf=True)

    def test_invert_vp_weights(self):
        self.actualSetUp(zerow=True, block=True)
        self._invert_base(name='invert_vp_weights', positionthreshold=2.0, check_components=False, grid_weights=True)
        
        maximum_ok = True
        for pol in range(4):
            pol_name = self.dirty[0].polarisation_frame.names[pol]
            maxval = numpy.max(self.dirty[0].data[:, pol])
            if abs(maxval - 1.0) > 1e-2:
                print("Weight %.6f in pol %s peak significantly different from unity " % (maxval, pol_name))
                maximum_ok = False
            else:
                print("Weight %.6f in pol %s sufficiently close to unity " % (maxval, pol_name))

        minimum_ok = True
        for pol in range(4):
            pol_name = self.dirty[0].polarisation_frame.names[pol]
            minval = numpy.min(self.dirty[0].data[:, pol])
            if minval < -1e-2:
                print("Weight %.6f in pol %s minimum signficantly negative " % (minval, pol_name))
                minimum_ok = False
            else:
                print("Weight %.6f in pol %s sufficient " % (minval, pol_name))

        assert maximum_ok, "Some weights have significant deviations from unit peak"
        assert minimum_ok, "Some weights have significant minimums"

if __name__ == '__main__':
    unittest.main()
