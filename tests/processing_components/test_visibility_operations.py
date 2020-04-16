""" Unit tests for visibility operations


"""

import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from rascil.data_models.memory_data_models import Skycomponent
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.imaging import dft_skycomponent_visibility
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import copy_visibility, create_visibility, create_blockvisibility, \
    create_visibility_from_rows, phaserotate_visibility
from rascil.processing_components.visibility.coalesce import convert_blockvisibility_to_visibility
from rascil.processing_components.visibility.operations import append_visibility, qa_visibility, \
    subtract_visibility, divide_visibility


class TestVisibilityOperations(unittest.TestCase):
    def setUp(self):
        self.lowcore = create_named_configuration('LOWBD2-CORE')
        self.times = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
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

    def test_create_blockvisibility(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          weight=1.0)
        assert self.vis.nvis == len(self.vis.time)

    def test_create_visibility1(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre,
                                     weight=1.0)
        assert self.vis.nvis == len(self.vis.time)
        assert self.vis.nvis == len(self.vis.frequency)

    def test_create_visibility_polarisation(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("linear"))
        assert self.vis.nvis == len(self.vis.time)
        assert self.vis.nvis == len(self.vis.frequency)

    def test_create_visibility_from_rows1(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0)
        rows = self.vis.time > 150.0
        for makecopy in [True, False]:
            selected_vis = create_visibility_from_rows(self.vis, rows, makecopy=makecopy)
            assert selected_vis.nvis == numpy.sum(numpy.array(rows))

    def test_create_visibility_time(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, channel_bandwidth=self.channel_bandwidth)
        assert self.vis.nvis == len(self.vis.time)

    def test_convert_blockvisibility(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                          weight=1.0, channel_bandwidth=self.channel_bandwidth)
        vis = convert_blockvisibility_to_visibility(self.vis)
        assert vis.nvis == len(vis.time)
        assert numpy.unique(vis.time).size == self.vis.time.size  # pylint: disable=no-member

    def test_create_visibility_from_rows_makecopy(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency, phasecentre=self.phasecentre,
                                     weight=1.0, channel_bandwidth=self.channel_bandwidth)
        rows = self.vis.time > 150.0
        for makecopy in [True, False]:
            selected_vis = create_visibility_from_rows(self.vis, rows, makecopy=makecopy)
            assert selected_vis.nvis == numpy.sum(numpy.array(rows))

    def test_append_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre,
                                     weight=1.0)
        othertimes = (numpy.pi / 43200.0) * numpy.arange(300.0, 600.0, 30.0)
        self.othervis = create_visibility(self.lowcore, othertimes, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          weight=1.0)
        self.vis = append_visibility(self.vis, self.othervis)
        assert self.vis.nvis == len(self.vis.time)
        assert self.vis.nvis == len(self.vis.frequency)

    def test_divide_visibility(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame("stokesI"))
        self.vis.data['vis'][..., :] = [2.0 + 0.0j]
        self.othervis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                               channel_bandwidth=self.channel_bandwidth,
                                               phasecentre=self.phasecentre,
                                               weight=1.0, polarisation_frame=PolarisationFrame("stokesI"))
        self.othervis.data['vis'][..., :] = [1.0 + 0.0j]
        self.ratiovis = divide_visibility(self.vis, self.othervis)
        assert self.ratiovis.nvis == self.vis.nvis
        assert numpy.max(numpy.abs(self.ratiovis.vis)) == 2.0, numpy.max(numpy.abs(self.ratiovis.vis))

    def test_divide_visibility_pol(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame("linear"))
        self.vis.data['vis'][..., :] = [2.0 + 0.0j, 0.0j, 0.0j, 2.0 + 0.0j]
        self.othervis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                               channel_bandwidth=self.channel_bandwidth,
                                               phasecentre=self.phasecentre,
                                               weight=1.0, polarisation_frame=PolarisationFrame("linear"))
        self.othervis.data['vis'][..., :] = [1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j]
        self.ratiovis = divide_visibility(self.vis, self.othervis)
        assert self.ratiovis.nvis == self.vis.nvis
        assert numpy.max(numpy.abs(self.ratiovis.vis)) == 2.0, numpy.max(numpy.abs(self.ratiovis.vis))

    def test_divide_visibility_singular(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre,
                                          weight=1.0, polarisation_frame=PolarisationFrame("linear"))
        self.vis.data['vis'][..., :] = [2.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j]
        self.othervis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                               channel_bandwidth=self.channel_bandwidth,
                                               phasecentre=self.phasecentre,
                                               weight=1.0, polarisation_frame=PolarisationFrame("linear"))
        self.othervis.data['vis'][..., :] = [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]
        self.ratiovis = divide_visibility(self.vis, self.othervis)
        assert self.ratiovis.nvis == self.vis.nvis
        assert numpy.max(numpy.abs(self.ratiovis.vis)) == 2.0, numpy.max(numpy.abs(self.ratiovis.vis))

    def test_copy_visibility(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth, phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        vis = copy_visibility(self.vis)
        self.vis.data['vis'] = 0.0
        vis.data['vis'] = 1.0
        assert (vis.data['vis'][0, 0].real == 1.0)
        assert (self.vis.data['vis'][0, 0].real == 0.0)

    def test_phase_rotation_identity(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
        newphasecenters = [SkyCoord(182, -35, unit=u.deg), SkyCoord(182, -30, unit=u.deg),
                           SkyCoord(177, -30, unit=u.deg), SkyCoord(176, -35, unit=u.deg),
                           SkyCoord(216, -35, unit=u.deg), SkyCoord(180, -70, unit=u.deg)]
        for newphasecentre in newphasecenters:
            # Phase rotating back should not make a difference
            original_vis = self.vismodel.vis
            original_uvw = self.vismodel.uvw
            rotatedvis = phaserotate_visibility(phaserotate_visibility(self.vismodel, newphasecentre, tangent=False),
                                                self.phasecentre, tangent=False)
            assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
            assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)

    def test_phase_rotation(self):
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

    def test_phase_rotation_block(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
        # Predict visibilities with new phase centre independently
        ha_diff = -(self.compabsdirection.ra - self.phasecentre.ra).to(u.rad).value
        vispred = create_blockvisibility(self.lowcore, self.times + ha_diff, self.frequency,
                                         channel_bandwidth=self.channel_bandwidth,
                                         phasecentre=self.compabsdirection, weight=1.0,
                                         polarisation_frame=PolarisationFrame("stokesIQUV"))
        vismodel2 = dft_skycomponent_visibility(vispred, self.comp)

        # Should yield the same results as rotation
        rotatedvis = phaserotate_visibility(self.vismodel, newphasecentre=self.compabsdirection, tangent=False)
        assert_allclose(rotatedvis.vis, vismodel2.vis, rtol=3e-6)
        assert_allclose(rotatedvis.uvw, vismodel2.uvw, rtol=3e-6)

    def test_phase_rotation_inverse(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
        there = SkyCoord(ra=+250.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        # Phase rotating back should not make a difference
        original_vis = self.vismodel.vis
        original_uvw = self.vismodel.uvw
        rotatedvis = phaserotate_visibility(phaserotate_visibility(self.vismodel, there, tangent=False,
                                                                   inverse=True),
                                            self.phasecentre, tangent=False, inverse=True)
        assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
        assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)

    def test_phase_rotation_inverse_block(self):
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
        there = SkyCoord(ra=+250.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        # Phase rotating back should not make a difference
        original_vis = self.vismodel.vis
        original_uvw = self.vismodel.uvw
        rotatedvis = phaserotate_visibility(phaserotate_visibility(self.vismodel, there, tangent=False,
                                                                   inverse=True),
                                            self.phasecentre, tangent=False, inverse=True)
        assert_allclose(rotatedvis.uvw, original_uvw, rtol=1e-7)
        assert_allclose(rotatedvis.vis, original_vis, rtol=1e-7)

    def test_subtract(self):
        vis1 = create_visibility(self.lowcore, self.times, self.frequency,
                                 channel_bandwidth=self.channel_bandwidth,
                                 phasecentre=self.phasecentre, weight=1.0,
                                 polarisation_frame=PolarisationFrame("stokesIQUV"))
        vis1.data['vis'][...] = 1.0
        vis2 = create_visibility(self.lowcore, self.times, self.frequency,
                                 channel_bandwidth=self.channel_bandwidth,
                                 phasecentre=self.phasecentre, weight=1.0,
                                 polarisation_frame=PolarisationFrame("stokesIQUV"))
        vis2.data['vis'][...] = 1.0
        zerovis = subtract_visibility(vis1, vis2)
        qa = qa_visibility(zerovis, context='test_qa')
        self.assertAlmostEqual(qa.data['maxabs'], 0.0, 7)

    def test_qa(self):
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
        self.vismodel = dft_skycomponent_visibility(self.vis, self.comp)
        qa = qa_visibility(self.vis, context='test_qa')
        self.assertAlmostEqual(qa.data['maxabs'], 100.0, 7)
        self.assertAlmostEqual(qa.data['medianabs'], 11.0, 7)
        assert qa.context == 'test_qa'

    def test_elevation(self):
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=+15.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = (numpy.pi / 43200.0) * numpy.arange(-43200, +43200, 3600.0)
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"),
                                     elevation_limit=numpy.pi * 15.0 / 180.0)
        n_elevation_limit = len(numpy.unique(self.vis.time))
        self.vis = create_visibility(self.lowcore, self.times, self.frequency,
                                     channel_bandwidth=self.channel_bandwidth,
                                     phasecentre=self.phasecentre, weight=1.0,
                                     polarisation_frame=PolarisationFrame("stokesIQUV"),
                                     elevation_limit=None)
        assert len(numpy.unique(self.vis.time)) >= n_elevation_limit

    def test_elevation_block(self):
        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=+15.0 * u.deg, frame='icrs', equinox='J2000')
        self.times = (numpy.pi / 43200.0) * numpy.arange(-43200, +43200, 3600.0)
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame("stokesIQUV"),
                                          elevation_limit=numpy.pi * 15.0 / 180.0)
        n_elevation_limit = len(numpy.unique(self.vis.time))
        self.vis = create_blockvisibility(self.lowcore, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame("stokesIQUV"),
                                          elevation_limit=None)
        assert len(numpy.unique(self.vis.time)) >= n_elevation_limit


if __name__ == '__main__':
    unittest.main()
