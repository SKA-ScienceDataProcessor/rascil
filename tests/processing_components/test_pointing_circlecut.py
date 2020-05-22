""" Unit tests for pointing

"""

import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.skycomponent.operations import create_skycomponent
from rascil.processing_components.calibration.pointing import create_pointingtable_from_blockvisibility
from rascil.processing_components.imaging.primary_beams import create_vp
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.pointing import simulate_gaintable_from_pointingtable
from rascil.processing_components.simulation import create_test_image
from rascil.processing_components.simulation.pointing import simulate_pointingtable
from rascil.processing_components.simulation import create_test_skycomponents_from_s3
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components import create_image

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestPointing(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path, rascil_data_path
        self.doplot = True
        
        self.midcore = create_named_configuration('MID', rmax=300.0)
        self.nants = len(self.midcore.names)
        self.dir = rascil_path('test_results')
        self.ntimes = 301
        self.times = numpy.linspace(-6.0, 6.0, self.ntimes) * numpy.pi / (12.0)
        
        self.frequency = numpy.array([1.4e9])
        self.channel_bandwidth = numpy.array([1e7])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-50.0 * u.deg, frame='icrs', equinox='J2000')
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0
        
        # Create model
        self.model = create_image(npixel=2048, cellsize=0.0003, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)

    def test_create_gaintable_from_pointingtable_circlecut_stokesI(self):
        self.sidelobe = SkyCoord(ra=+15.0 * u.deg, dec=-49.4 * u.deg, frame='icrs', equinox='J2000')
        comp = create_skycomponent(direction=self.sidelobe, flux=[[1.0]], frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesI'))
    
        telescopes = ['MID', 'MID_GAUSS']
        for telescope in telescopes:
            pt = create_pointingtable_from_blockvisibility(self.vis)
            pt = simulate_pointingtable(pt, pointing_error=0.0,
                                        global_pointing_error=[0.0, 0.0])
            vp = create_vp(self.model, telescope)
            gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
            if self.doplot:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.plot(gt[0].time, numpy.real(gt[0].gain[:, 0, 0, 0, 0]), '.', label='stokesI Real')
                plt.plot(gt[0].time, numpy.imag(gt[0].gain[:, 0, 0, 0, 0]), '.', label='stokesI Imaginary')
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Gain')
                plt.title('test_create_gaintable_from_pointingtable_%s' % telescope)
                plt.show(block=False)
            assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 1, 1), gt[0].gain.shape

    def test_create_gaintable_from_pointingtable_circlecut_stokesIQUV(self):
        self.vis = create_blockvisibility(self.midcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('linear'))
        self.sidelobe = SkyCoord(ra=+15.0 * u.deg, dec=-49.3 * u.deg, frame='icrs', equinox='J2000')
        comp = create_skycomponent(direction=self.sidelobe, flux=[[1.0, 0.0, 0.0, 0.0]],
                                   frequency=self.frequency,
                                   polarisation_frame=PolarisationFrame('stokesIQUV'))
    
        telescopes = ['MID_FEKO_B2']
        for telescope in telescopes:
            pt = create_pointingtable_from_blockvisibility(self.vis)
            pt = simulate_pointingtable(pt, pointing_error=0.0,
                                        global_pointing_error=[0.0, 0.0])
            vp = create_vp(self.model, telescope)
            gt = simulate_gaintable_from_pointingtable(self.vis, [comp], pt, vp)
            if self.doplot:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.plot(gt[0].time, numpy.real(gt[0].gain[:, 0, 0, 0, 0]), '.', label='XX')
                plt.plot(gt[0].time, numpy.real(gt[0].gain[:, 0, 0, 1, 1]), '.', label='YY')
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Gain')
                plt.title('test_create_gaintable_from_pointingtable_parallel_%s' % telescope)
                plt.show(block=False)
                plt.clf()
                plt.plot(gt[0].time, numpy.real(gt[0].gain[:, 0, 0, 0, 1]), '.', label='XY')
                plt.plot(gt[0].time, numpy.real(gt[0].gain[:, 0, 0, 1, 0]), '.', label='YX')
                plt.legend()
                plt.xlabel('Time (s)')
                plt.ylabel('Gain')
                plt.title('test_create_gaintable_from_pointingtable_cross_%s' % telescope)
                plt.show(block=False)

            assert gt[0].gain.shape == (self.ntimes, self.nants, 1, 2, 2), gt[0].gain.shape


if __name__ == '__main__':
    unittest.main()
