""" Unit tests for mpc

"""
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_image, create_empty_image_like
from rascil.processing_components.image.operations import import_image_from_fits, export_image_to_fits
from rascil.processing_components.imaging.primary_beams import create_low_test_beam
from rascil.processing_components.simulation import create_low_test_skycomponents_from_gleam, \
    create_test_skycomponents_from_s3
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation import create_test_image
from rascil.processing_components.simulation.atmospheric_screen import create_gaintable_from_screen, \
    grid_gaintable_to_screen, plot_gaintable_on_screen
from rascil.processing_components.skycomponent.operations import apply_beam_to_skycomponent
from rascil.processing_components.skycomponent.operations import filter_skycomponents_by_flux
from rascil.processing_components.visibility.base import create_blockvisibility

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestAtmosphericScreen(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("RASCIL_PERSIST", False)
        self.dir = rascil_path('test_results')

    def actualSetup(self, atmosphere="ionosphere"):
        dec = -40.0 * u.deg
        self.times = numpy.linspace(-10.0, 10.0, 3) * numpy.pi / (3600.0 * 12.0)
        self.phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=dec, frame='icrs', equinox='J2000')

        if atmosphere == "ionosphere":
            self.core = create_named_configuration('LOWBD2', rmax=300.0)
            self.frequency = numpy.array([1.0e8])
            self.channel_bandwidth = numpy.array([5e7])
            self.cellsize = 0.000015
        else:
            self.core = create_named_configuration('MID', rmax=300.0)
            self.frequency = numpy.array([1.36e9])
            self.channel_bandwidth = numpy.array([1e8])
            self.cellsize = 0.00015

        self.vis = create_blockvisibility(self.core, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=PolarisationFrame('stokesI'))
        self.vis.data['vis'] *= 0.0

        # Create model
        self.model = create_image(npixel=512, cellsize=0.000015, polarisation_frame=PolarisationFrame("stokesI"),
                                  frequency=self.frequency, channel_bandwidth=self.channel_bandwidth,
                                  phasecentre=self.phasecentre)

    def test_read_screen(self):
        screen = import_image_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        assert screen.data.shape == (1, 3, 2000, 2000), screen.data.shape

    def test_create_gaintable_from_screen_ionosphere(self):
        self.actualSetup("ionosphere")
        screen = import_image_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        beam = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                 frequency=self.frequency)

        beam = create_low_test_beam(beam, use_local=False)

        gleam_components = \
            create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                     phasecentre=self.phasecentre,
                                                     frequency=self.frequency,
                                                     polarisation_frame=PolarisationFrame('stokesI'),
                                                     radius=0.2)

        pb_gleam_components = apply_beam_to_skycomponent(gleam_components, beam)

        actual_components = filter_skycomponents_by_flux(pb_gleam_components, flux_min=1.0)

        gaintables = create_gaintable_from_screen(self.vis, actual_components, screen)
        assert len(gaintables) == len(actual_components), len(gaintables)
        assert gaintables[0].gain.shape == (3, 94, 1, 1, 1), gaintables[0].gain.shape

    def test_create_gaintable_from_screen_troposphere(self):
        self.actualSetup("troposphere")
        screen = import_image_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        beam = create_test_image(cellsize=0.00015, phasecentre=self.vis.phasecentre,
                                 frequency=self.frequency)

        beam = create_low_test_beam(beam, use_local=False)

        s3_components = create_test_skycomponents_from_s3(flux_limit=0.3,
                                                          phasecentre=self.phasecentre,
                                                          frequency=self.frequency,
                                                          polarisation_frame=PolarisationFrame('stokesI'),
                                                          radius=1.5 * numpy.pi / 180.0)

        assert len(s3_components) > 0, "No S3 components selected"

        pb_s3_components = apply_beam_to_skycomponent(s3_components, beam)

        actual_components = filter_skycomponents_by_flux(pb_s3_components, flux_max=10.0)

        assert len(actual_components) > 0, "No components after applying primary beam"

        gaintables = create_gaintable_from_screen(self.vis, actual_components, screen, height=3e3,
                                                  type_atmosphere="troposphere")
        assert len(gaintables) == len(actual_components), len(gaintables)
        assert gaintables[0].gain.shape == (3, 63, 1, 1, 1), gaintables[0].gain.shape

    def test_grid_gaintable_to_screen(self):
        self.actualSetup()
        screen = import_image_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        beam = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                 frequency=self.frequency)

        beam = create_low_test_beam(beam, use_local=False)

        gleam_components = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                                    phasecentre=self.phasecentre,
                                                                    frequency=self.frequency,
                                                                    polarisation_frame=PolarisationFrame(
                                                                        'stokesI'),
                                                                    radius=0.2)

        pb_gleam_components = apply_beam_to_skycomponent(gleam_components, beam)

        actual_components = filter_skycomponents_by_flux(pb_gleam_components, flux_min=1.0)

        gaintables = create_gaintable_from_screen(self.vis, actual_components, screen)
        assert len(gaintables) == len(actual_components), len(gaintables)
        assert gaintables[0].gain.shape == (3, 94, 1, 1, 1), gaintables[0].gain.shape

        newscreen = create_empty_image_like(screen)

        newscreen, weights = grid_gaintable_to_screen(self.vis, gaintables, newscreen)
        assert numpy.max(numpy.abs(screen.data)) > 0.0
        if self.persist: export_image_to_fits(newscreen, rascil_path('test_results/test_mpc_screen_gridded.fits'))
        if self.persist: export_image_to_fits(weights, rascil_path('test_results/test_mpc_screen_gridded_weights.fits'))

    def test_plot_gaintable_to_screen(self):
        self.actualSetup()
        screen = import_image_from_fits(rascil_data_path('models/test_mpc_screen.fits'))
        beam = create_test_image(cellsize=0.0015, phasecentre=self.vis.phasecentre,
                                 frequency=self.frequency)

        beam = create_low_test_beam(beam, use_local=False)

        gleam_components = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                                    phasecentre=self.phasecentre,
                                                                    frequency=self.frequency,
                                                                    polarisation_frame=PolarisationFrame(
                                                                        'stokesI'),
                                                                    radius=0.2)

        pb_gleam_components = apply_beam_to_skycomponent(gleam_components, beam)

        actual_components = filter_skycomponents_by_flux(pb_gleam_components, flux_min=1.0)

        gaintables = create_gaintable_from_screen(self.vis, actual_components, screen)
        assert len(gaintables) == len(actual_components), len(gaintables)
        assert gaintables[0].gain.shape == (3, 94, 1, 1, 1), gaintables[0].gain.shape

        plot_gaintable_on_screen(self.vis, gaintables, plotfile=rascil_path(
            'test_results/test_plot_gaintable_to_screen.png'))
