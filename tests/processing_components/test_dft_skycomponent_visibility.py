""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility, idft_visibility_skycomponent
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_visibility, create_blockvisibility, \
    phaserotate_visibility


class TestVisibilityDFTOperations(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2', rmax=300.0)
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 100.0)
        self.frequency = numpy.linspace(1.0e8, 1.1e8, 3)
        self.channel_bandwidth = numpy.array([1e7, 1e7, 1e7])
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0, 20.0, -10.0, 1.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux)

    def test_phase_rotation_stokesi(self):
        # Define the component and give it some spectral behaviour
        f = numpy.array([100.0])
        self.flux = numpy.array([f, 0.8 * f, 0.6 * f])

        # The phase centre is absolute and the component is specified relative (for now).
        # This means that the component should end up at the position phasecentre+compredirection
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        self.compabsdirection = SkyCoord(ra=+181.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        pcof = self.phasecentre.skyoffset_frame()
        self.compreldirection = self.compabsdirection.transform_to(pcof)
        self.comp = Skycomponent(direction=self.compreldirection, frequency=self.frequency, flux=self.flux,
                                 polarisation_frame=PolarisationFrame("stokesI"))

        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesI"))
        self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
        # Predict visibilities with new phase centre independently
        ha_diff = -(self.compabsdirection.ra - self.phasecentre.ra).to(u.rad).value
        vispred = create_visibility(self.lowcore, self.times + ha_diff, self.frequency,
                                    channel_bandwidth=self.channel_bandwidth,
                                    phasecentre=self.compabsdirection, weight=1.0,
                                    polarisation_frame=PolarisationFrame("stokesI"))
        vismodel2 = dft_skycomponent_visibility(vispred, self.comp)

        # Should yield the same results as rotation
        rotatedvis = phaserotate_visibility(self.vismodel, newphasecentre=self.compabsdirection, tangent=False)
        assert_allclose(rotatedvis.vis, vismodel2.vis, rtol=3e-6)
        assert_allclose(rotatedvis.uvw, vismodel2.uvw, rtol=3e-6)

    def test_phase_rotation_stokesiquv(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
        # Predict visibilities with new phase centre independently
        ha_diff = -(self.compabsdirection.ra - self.phasecentre.ra).to(u.rad).value
        vispred = create_visibility(self.lowcore, self.times + ha_diff, self.frequency,
                                    channel_bandwidth=self.channel_bandwidth,
                                    phasecentre=self.compabsdirection, weight=1.0,
                                    polarisation_frame=PolarisationFrame("stokesIQUV"))
        vismodel2 = dft_skycomponent_visibility(vispred, self.comp)

        # Should yield the same results as rotation
        rotatedvis = phaserotate_visibility(self.vismodel, newphasecentre=self.compabsdirection, tangent=False)
        assert_allclose(rotatedvis.vis, vismodel2.vis, rtol=3e-6)
        assert_allclose(rotatedvis.uvw, vismodel2.uvw, rtol=3e-6)

    def test_dft_idft_stokesiquv_blockvisibility(self):
        for vpol in [PolarisationFrame("linear"), PolarisationFrame("circular")]:
            self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                              channel_bandwidth=self.channel_bandwidth,
                                              phasecentre=self.phasecentre, weight=1.0,
                                              polarisation_frame=PolarisationFrame("stokesIQUV"))
            self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
            rcomp, weights = idft_visibility_skycomponent(self.vismodel, self.comp)
            assert_allclose(self.comp.flux, numpy.real(rcomp[0].flux), rtol=1e-11)

    def test_dft_idft_stokesiquv_visibility(self):
        for vpol in [PolarisationFrame("linear"), PolarisationFrame("circular")]:
            self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                         channel_bandwidth=self.channel_bandwidth,
                                         phasecentre=self.phasecentre, weight=1.0,
                                         polarisation_frame=PolarisationFrame("stokesIQUV"))
            self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
            rcomp, weights = idft_visibility_skycomponent(self.vismodel, self.comp)
            assert_allclose(self.comp.flux, numpy.real(rcomp[0].flux), rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
