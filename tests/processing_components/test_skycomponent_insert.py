""" Unit tests for skycomponents

"""
import os
import logging
import unittest
from numpy.testing import assert_array_almost_equal


import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.image.operations import export_image_to_fits, create_image
from rascil.processing_components.imaging.base import predict_2d, invert_2d
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.skycomponent.operations import insert_skycomponent, create_skycomponent
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_visibility

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestSkycomponentInsert(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("RASCIL_PERSIST", False)

        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.dir = rascil_path('test_results')
        
    def actualSetup(self, dopol=False):
        
        if dopol:
            self.vis_pol = PolarisationFrame("linear")
            self.image_pol = PolarisationFrame("stokesIQUV")
            self.pol_flux = numpy.array([1.0, -0.8, 0.2, 0.01])
        else:
            self.vis_pol = PolarisationFrame("stokesI")
            self.image_pol = PolarisationFrame("stokesI")
            self.pol_flux = numpy.array([1.0])
        
        self.times = (numpy.pi / 12.0) * numpy.linspace(-3.0, 3.0, 7)
        self.image_frequency = numpy.linspace(0.9e8, 1.1e8, 5)
        self.image_channel_bandwidth = numpy.array(5*[5e6])
        self.component_frequency = numpy.linspace(0.8e8, 1.2e8, 7)
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_visibility(self.lowcore, self.times, self.image_frequency,
                                     channel_bandwidth=self.image_channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=self.vis_pol, zerow=True)
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=256, cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                  frequency=self.image_frequency, channel_bandwidth=self.image_channel_bandwidth,
                                  polarisation_frame=self.image_pol)

        dphasecentre = SkyCoord(ra=+181.0 * u.deg, dec=-58.0 * u.deg, frame='icrs', equinox='J2000')
        flux_scale = numpy.power(self.component_frequency/1e8, -0.7)
        self.flux = numpy.outer(flux_scale, self.pol_flux)
        self.sc = create_skycomponent(direction=dphasecentre, flux=self.flux,
                                    frequency=self.component_frequency,
                                    polarisation_frame=self.image_pol)

    def test_insert_skycomponent_FFT(self):
    
        self.actualSetup()
    
        self.sc = create_skycomponent(direction=self.phasecentre, flux=self.sc.flux,
                                      frequency=self.component_frequency,
                                      polarisation_frame=self.image_pol)
    
        insert_skycomponent(self.model, self.sc)
        npixel = self.model.shape[3]
        # WCS is 1-relative
        rpix = numpy.round(self.model.wcs.wcs.crpix).astype('int') - 1
        assert rpix[0] == npixel // 2
        assert rpix[1] == npixel // 2
        # The phase centre is at rpix[0], rpix[1] in 0-relative pixels
        assert self.model.data[2, 0, rpix[1], rpix[0]] == self.pol_flux[0]
        # If we predict the visibility, then the imaginary part must be zero. This is determined entirely
        # by shift_vis_to_image in processing_components.imaging.base
        self.vis.data['vis'][...] = 0.0
        self.vis = predict_2d(self.vis, self.model)
        # The actual phase centre of a numpy FFT is at nx //2, nx //2 (0 rel).
        assert numpy.max(numpy.abs(self.vis.vis.imag)) < 1e-3

    def test_insert_skycomponent_FFT_IQUV(self):
    
        self.actualSetup(dopol=True)
    
        self.sc = create_skycomponent(direction=self.phasecentre, flux=self.sc.flux,
                                      frequency=self.component_frequency,
                                      polarisation_frame=self.image_pol)
    
        insert_skycomponent(self.model, self.sc)
        npixel = self.model.shape[3]
        # WCS is 1-relative
        rpix = numpy.round(self.model.wcs.wcs.crpix).astype('int') - 1
        assert rpix[0] == npixel // 2
        assert rpix[1] == npixel // 2
        # The phase centre is at rpix[0], rpix[1] in 0-relative pixels
        assert_array_almost_equal(self.model.data[2, :, rpix[1], rpix[0]], self.flux[3, :], 8)
        
        # If we predict the visibility, then the imaginary part must be zero. This is determined entirely
        # by shift_vis_to_image in processing_components.imaging.base
        self.vis.data['vis'][...] = 0.0
        self.vis = predict_2d(self.vis, self.model)
        # The actual phase centre of a numpy FFT is at nx //2, nx //2 (0 rel).

        assert numpy.max(numpy.abs(self.vis.vis[...,0].imag)) == 0.0
        assert numpy.max(numpy.abs(self.vis.vis[...,3].imag)) == 0.0

    def test_insert_skycomponent_dft(self):
        self.actualSetup()

        self.sc = create_skycomponent(direction=self.phasecentre, flux=self.sc.flux,
                                    frequency=self.component_frequency,
                                    polarisation_frame=PolarisationFrame('stokesI'))

        self.vis.data['vis'][...] = 0.0
        self.vis = dft_skycomponent_visibility(self.vis, self.sc)
        im, sumwt = invert_2d(self.vis, self.model)
        if self.persist: export_image_to_fits(im, '%s/test_skycomponent_dft.fits' % self.dir)
        assert numpy.max(numpy.abs(self.vis.vis.imag)) < 1e-3

    def test_insert_skycomponent_nearest(self):
        self.actualSetup()
    
        insert_skycomponent(self.model, self.sc, insert_method='Nearest')
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 1.0, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.0, 7)

    def test_insert_skycomponent_nearest_IQUV(self):
        self.actualSetup(dopol=True)
    
        insert_skycomponent(self.model, self.sc, insert_method='Nearest')
        # These test a regression but are not known a priori to be correct
        for pol in range(4):
            self.assertAlmostEqual(self.model.data[2, pol, 151, 122], self.pol_flux[pol], 7)
            self.assertAlmostEqual(self.model.data[2, pol, 152, 122], 0.0, 7)

    def test_insert_skycomponent_sinc(self):
        self.actualSetup()
    
        insert_skycomponent(self.model, self.sc, insert_method='Sinc')
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.87684398703184396, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.2469311811046056, 7)
    
    def test_insert_skycomponent_sinc_bandwidth(self):
        self.actualSetup()
    
        insert_skycomponent(self.model, self.sc, insert_method='Sinc', bandwidth=0.5)
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.25133066186805758, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.19685222464041874, 7)
    
    def test_insert_skycomponent_lanczos(self):
        self.actualSetup()
    
        insert_skycomponent(self.model, self.sc, insert_method='Lanczos')
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.87781267543090036, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.23817562762032077, 7)

    def test_insert_skycomponent_lanczos_IQUV(self):
        self.actualSetup(dopol=True)
    
        insert_skycomponent(self.model, self.sc, insert_method='Lanczos')
        # These test a regression but are not known a priori to be correct
        for pol in range(4):
            self.assertAlmostEqual(self.model.data[2, pol, 151, 122], self.pol_flux[pol] * 0.87781267543090036, 7)
            self.assertAlmostEqual(self.model.data[2, pol, 152, 122], self.pol_flux[pol] * 0.23817562762032077, 7)

    def test_insert_skycomponent_lanczos_bandwidth(self):
        self.actualSetup()
    
        insert_skycomponent(self.model, self.sc, insert_method='Lanczos', bandwidth=0.5)
        # These test a regression but are not known a priori to be correct
        self.assertAlmostEqual(self.model.data[2, 0, 151, 122], 0.24031092091707615, 7)
        self.assertAlmostEqual(self.model.data[2, 0, 152, 122], 0.18648989466050975, 7)


if __name__ == '__main__':
    unittest.main()
