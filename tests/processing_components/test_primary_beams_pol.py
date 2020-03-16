"""Unit tests for primary beam application with polarisation


"""

import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame, convert_pol_frame
from rascil.processing_components.imaging.base import create_image_from_visibility
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility, idft_visibility_skycomponent
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.skycomponent import create_skycomponent, apply_beam_to_skycomponent, \
    copy_skycomponent
from rascil.processing_components.visibility import create_blockvisibility, sum_visibility

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestPrimaryBeamsPol(unittest.TestCase):
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", False)
    
    def createVis(self, config='MID', dec=-35.0, rmax=1e2, freq=1.3e9):
        self.frequency = [freq]
        self.channel_bandwidth = [1e6]
        self.flux = numpy.array([[100.0, 60.0, -10.0, +1.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs',
                                    equinox='J2000')
        self.config = create_named_configuration(config)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        self.npixel = 1024
        self.fov = 8
        self.cellsize = numpy.pi * self.fov / (self.npixel * 180.0)
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants
        
        self.config = create_named_configuration(config, rmax=rmax)
        self.phasecentre = SkyCoord(ra=+15 * u.deg, dec=dec * u.deg, frame='icrs', equinox='J2000')
    
    def test_apply_primary_beam_imageplane(self):
        self.createVis()
        telescope = 'MID'
        flux = numpy.array([[100.0, 60.0, -10.0, +1.0]])
        ipol = PolarisationFrame("stokesIQUV")
        print("Original flux = {}".format(flux))
        apply_pb = False
        for vpol in (PolarisationFrame("linear"), PolarisationFrame("circular")):
            print("Testing {0}".format(vpol.type))
            bvis = create_blockvisibility(self.config, self.times, self.frequency,
                                          channel_bandwidth=self.channel_bandwidth,
                                          phasecentre=self.phasecentre, weight=1.0,
                                          polarisation_frame=vpol)
            
            component_centre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs',
                                        equinox='J2000')
            component = create_skycomponent(direction=component_centre, flux=flux,
                                            frequency=self.frequency,
                                            polarisation_frame=PolarisationFrame("stokesIQUV"))
            model = create_image_from_visibility(bvis, cellsize=self.cellsize, npixel=self.npixel,
                                                 override_cellsize=False,
                                                 polarisation_frame=PolarisationFrame("stokesIQUV"))
            beam = create_pb(model, telescope=telescope, use_local=False)
            if apply_pb:
                pbcomp = apply_beam_to_skycomponent(component, beam)
            else:
                pbcomp = copy_skycomponent(component)
            print(pbcomp.flux)
            bvis = dft_skycomponent_visibility(bvis, pbcomp)
            iquv_image = convert_pol_frame(sum_visibility(bvis, component.direction)[0], ipol, vpol, 1)
            print("IQUV to {0} to IQUV image = {1}".format(vpol.type, iquv_image))


if __name__ == '__main__':
    unittest.main()
