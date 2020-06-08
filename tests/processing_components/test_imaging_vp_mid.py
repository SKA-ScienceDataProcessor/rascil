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

from rascil.data_models.polarisation import PolarisationFrame, convert_linear_to_stokes
from rascil.processing_components import convert_azelvp_to_radec
from rascil.processing_components.griddata.kernels import create_vpterm_convolutionfunction
from rascil.processing_components.griddata import convert_convolutionfunction_to_image
from rascil.processing_components.image.operations import export_image_to_fits, smooth_image, fft_image, copy_image, \
    qa_image
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility
from rascil.processing_components.imaging.imaging_vp import predict_vp, invert_vp
from rascil.processing_components.imaging.weighting import weight_blockvisibility, taper_visibility_gaussian
from rascil.processing_components.imaging.primary_beams import create_vp, create_vp_generic
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import ingest_unittest_visibility, \
    create_unittest_model, create_unittest_components
from rascil.processing_components.skycomponent.operations import find_skycomponents, find_nearest_skycomponent, \
    insert_skycomponent, apply_voltage_pattern_to_skycomponent
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
    
    def actualSetUp(self, zerow=True, block=False, image_pol=PolarisationFrame("stokesIQUV"), npixel=None, rmax=300.0,
                    scale=0.25, cellsize=None, telescope="MID_IMPERFECT"):
        self.telescope = telescope
        self.doplot = False
        self.npixel = npixel
        self.low = create_named_configuration('MIDR5', rmax=rmax)
        self.freqwin = 1
        self.vis_list = list()
        self.ntimes = 21
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
            self.npixel = 512
            print("Npixel = ", self.npixel)
        if cellsize is None:
            self.cellsize = advice['cellsize']
            print("Cellsize = {:3f} arcsec".format(self.cellsize * 3600.0 * 180.0 / numpy.pi))

        self.model = create_unittest_model(self.vis, self.image_pol, cellsize=self.cellsize,
                                           npixel=self.npixel, nchan=self.freqwin)
        
        self.vis = weight_blockvisibility(self.vis, self.model)
        self.vis = taper_visibility_gaussian(self.vis, 2.0 * self.cellsize)
        self.components = create_unittest_components(self.model, flux, applypb=False,
                                                     scale=scale, single=False, symmetric=False)

        def find_vp_renormalised(model, renormalise=False):
            if telescope == "MID_FEKO_B2":
                vp = create_vp(telescope=telescope)
            elif telescope == "MID_IMPERFECT":
                vp = create_vp_generic(model, diameter=15.0, polarisation_frame=PolarisationFrame("linear"),
                                       no_cross_pol=True, cross_pol=[-0.01 + 0.03j, +0.002 - 0.006j])
            else:
                self.telescope = "MID_PERFECT"
                vp = create_vp_generic(model, diameter=15.0, polarisation_frame=PolarisationFrame("linear"),
                                       no_cross_pol=True)
            if renormalise:
                g = numpy.zeros([4])
                g[0] = numpy.max(numpy.abs(vp.data[:, 0, ...]))
                g[3] = numpy.max(numpy.abs(vp.data[:, 3, ...]))
                g[1] = g[2] = numpy.sqrt(g[0] * g[3])
                for pol in range(4):
                    vp.data[:, pol, ...] /= g[pol]
            vp = convert_azelvp_to_radec(vp, model, pa=0.0)
            return vp

        make_vp = functools.partial(find_vp_renormalised)
        self.vp = make_vp(self.model)
        print(telescope)
        export_image_to_fits(self.vp, "%s/test_imaging_vp_%s_mid_vp.fits" % (self.dir, self.telescope))

        self.pb_components = apply_voltage_pattern_to_skycomponent(self.components, self.vp)
        for comp in self.pb_components:
            comp.flux = convert_linear_to_stokes(comp.flux)
            comp.flux = comp.flux.real
        self.pb_model = copy_image(self.model)
        self.pb_model.data[...] = 0.0
        self.pb_model = insert_skycomponent(self.pb_model, self.pb_components)
        self.model = insert_skycomponent(self.model, self.components)
        self.vis = dft_skycomponent_visibility(self.vis, self.components)

        self.gcf, self.cf = create_vpterm_convolutionfunction(self.model,
                                                              make_vp=find_vp_renormalised,
                                                              oversampling=1,
                                                              support=16,
                                                              use_aaf=False)
        cf_image = convert_convolutionfunction_to_image(self.cf)
        cf_image.data = numpy.real(cf_image.data)
        export_image_to_fits(cf_image, "%s/test_imaging_vp_%s_mid_cf.fits" % (self.dir, self.telescope))
        
        self.pb_cmodel = smooth_image(self.pb_model)
        if self.persist:
            export_image_to_fits(self.model, '%s/test_imaging_vp_%s_model.fits' % (self.dir, self.telescope))
            export_image_to_fits(self.pb_model, '%s/test_imaging_vp_%s_pb_model.fits' % (self.dir, self.telescope))
            export_image_to_fits(self.pb_cmodel, '%s/test_imaging_vp_%s_pb_cmodel.fits' % (self.dir, self.telescope))
        self.peak = numpy.unravel_index(numpy.argmax(numpy.abs(self.pb_cmodel.data)), self.pb_cmodel.shape)
        
    def _checkcomponents(self, dirty, fluxthreshold=0.6, positionthreshold=0.1):
        comps = find_skycomponents(dirty, fwhm=1.0, threshold=fluxthreshold, npixels=5)
        
        for comp in comps:
            # Check for agreement in direction
            ocomp, separation = find_nearest_skycomponent(comp.direction, self.pb_components)
            assert separation / self.cellsize < positionthreshold, \
                "Component differs in position {} pixels".format(separation / self.cellsize)
            err = numpy.max(numpy.abs(comp.flux - ocomp.flux))
            print(comp.flux, ocomp.flux, comp.flux-ocomp.flux, err)
            assert err < fluxthreshold, "Component has significant error {}".format(comp.flux-ocomp.flux)
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
        
        if self.persist: export_image_to_fits(dirty[0], '%s/test_imaging_vp_%s_mid_%s_residual.fits' %
                                              (self.dir, self.telescope, name))
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
            export_image_to_fits(dirty[0], '%s/test_imaging_vp_%s_mid_%s_dirty.fits' % (self.dir, self.telescope, name))
            dirty_fft = fft_image(dirty[0])
            export_image_to_fits(dirty_fft, '%s/test_imaging_vp_%s_mid_%s_dirty_fft.fits' % (self.dir, self.telescope, name))

        if check_components:
            self._checkcomponents(dirty[0], fluxthreshold, positionthreshold)
    
    def test_predict_vp(self):
        for telescope in ["MID_PERFECT", "MID_IMPERFECT", "MID_FEKO_B2"]:
            self.actualSetUp(zerow=True, block=True, telescope=telescope)
            self._predict_base(name='predict_vp', fluxthreshold=0.2)
    
    def test_invert_vp(self):
        for telescope in ["MID_FEKO_B2", "MID_PERFECT", "MID_IMPERFECT"]:
            self.actualSetUp(zerow=True, block=True, telescope=telescope)
            self._invert_base(name='invert_vp', positionthreshold=2.0, check_components=True, fluxthreshold=10.0)

    def test_invert_vp_psf(self):
        for telescope in ["MID_PERFECT", "MID_IMPERFECT", "MID_FEKO_B2"]:
            self.actualSetUp(zerow=True, block=True, telescope=telescope)
            self._invert_base(name='invert_vp_psf', positionthreshold=2.0, check_components=False, fluxthreshold=1.0,
                            dopsf=True)

    def test_invert_vp_weights(self):
        for telescope in ["PERFECT", "IMPERFECT", "MID_FEKO_B2"]:
            self.actualSetUp(zerow=True, block=True, telescope=telescope)
            self._invert_base(name='invert_vp_weights', positionthreshold=2.0, check_components=False, grid_weights=True)
        
        maximum_ok = True
        for pol in range(4):
            pol_name = self.dirty[0].polarisation_frame.names[pol]
            maxval = numpy.max(self.dirty[0].data[:, pol])
            if abs(maxval - 1.0) > 2e-2:
                print("Weight %.6f in pol %s peak significantly different from unity " % (maxval, pol_name))
                maximum_ok = False
            else:
                print("Weight %.6f in pol %s sufficiently close to unity " % (maxval, pol_name))

        minimum_ok = True
        for pol in range(4):
            pol_name = self.dirty[0].polarisation_frame.names[pol]
            minval = numpy.min(self.dirty[0].data[:, pol])
            if minval < -2e-2:
                print("Weight %.6f in pol %s minimum signficantly negative " % (minval, pol_name))
                minimum_ok = False
            else:
                print("Weight %.6f in pol %s sufficient " % (minval, pol_name))

        assert maximum_ok, "Some weights have significant deviations from unit peak"
        assert minimum_ok, "Some weights have significant minimums"

if __name__ == '__main__':
    unittest.main()
