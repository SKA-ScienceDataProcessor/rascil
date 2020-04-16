""" Unit tests for visibility weighting
"""
import os
import logging
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils import fit_2dgaussian

from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components import fft_image

from rascil.processing_components.image.operations import export_image_to_fits
from rascil.processing_components.imaging.base import invert_2d
from rascil.processing_components.imaging.base import create_image_from_visibility
from rascil.processing_components.imaging.weighting import weight_visibility, taper_visibility_gaussian, taper_visibility_tukey
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_visibility

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestWeighting(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        self.npixel = 512

        self.persist = os.getenv("RASCIL_PERSIST", False)

    def actualSetUp(self, time=None, dospectral=False, image_pol=PolarisationFrame("stokesI")):
        self.lowcore = create_named_configuration('LOWBD2', rmax=600)
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 5)
        
        if time is not None:
            self.times = time
        log.info("Times are %s" % (self.times))
        
        if dospectral:
            self.nchan = 3
            self.frequency = numpy.array([0.9e8, 1e8, 1.1e8])
            self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        else:
            self.frequency = numpy.array([1e8])
            self.channel_bandwidth = numpy.array([1e7])
            
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

        if dospectral:
            numpy.array([f, 0.8 * f, 0.6 * f])
        else:
            numpy.array([f])

        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.componentvis = create_visibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre,
                                              weight=1.0, polarisation_frame=self.vis_pol)
        self.uvw = self.componentvis.data['uvw']
        self.componentvis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image_from_visibility(self.componentvis, npixel=self.npixel, cellsize=0.0005,
                                                  nchan=len(self.frequency),
                                                  polarisation_frame=self.image_pol)

    def test_tapering_Gaussian(self):
        self.actualSetUp()
        size_required = 0.010
        self.componentvis = weight_visibility(self.componentvis, self.model, algoritm='uniform')
        self.componentvis = taper_visibility_gaussian(self.componentvis, beam=size_required)
        psf, sumwt = invert_2d(self.componentvis, self.model, dopsf=True)
        if self.persist:
            export_image_to_fits(psf, '%s/test_weighting_gaussian_taper_psf.fits' % self.dir)
        xfr = fft_image(psf)
        xfr.data = xfr.data.real.astype('float')
        if self.persist:
            export_image_to_fits(xfr, '%s/test_weighting_gaussian_taper_xfr.fits' % self.dir)
        npixel = psf.data.shape[3]
        sl = slice(npixel // 2 - 7, npixel // 2 + 8)
        fit = fit_2dgaussian(psf.data[0, 0, sl, sl])
        # if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
        #     raise ValueError('Error in fitting to psf')
        # fit_2dgaussian returns sqrt of variance. We need to convert that to FWHM.
        # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        scale_factor = numpy.sqrt(8 * numpy.log(2.0))
        size = numpy.sqrt(fit.x_stddev * fit.y_stddev) * scale_factor
        # Now we need to convert to radians
        size *= numpy.pi * self.model.wcs.wcs.cdelt[1] / 180.0
        # Very impressive! Desired 0.01 Acheived 0.0100006250829
        assert numpy.abs(size - size_required) < 0.03 * size_required, \
            "Fit should be %f, actually is %f" % (size_required, size)

    def test_tapering_Tukey(self):
        self.actualSetUp()
        self.componentvis = weight_visibility(self.componentvis, self.model, algoritm='uniform')
        self.componentvis = taper_visibility_tukey(self.componentvis, tukey=1.0)
        psf, sumwt = invert_2d(self.componentvis, self.model, dopsf=True)
        if self.persist:
            export_image_to_fits(psf, '%s/test_weighting_tukey_taper_psf.fits' % self.dir)
        xfr = fft_image(psf)
        xfr.data = xfr.data.real.astype('float')
        if self.persist:
            export_image_to_fits(xfr, '%s/test_weighting_tukey_taper_xfr.fits' % self.dir)


if __name__ == '__main__':
    unittest.main()
