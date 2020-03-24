""" Unit tests for conversion of kernels to list


"""
import functools
import logging
import os
import unittest

import astropy.units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_image
from rascil.processing_components.griddata import convert_kernel_to_list
from rascil.processing_components.griddata.kernels import create_awterm_convolutionfunction
from rascil.processing_components.imaging.primary_beams import create_pb_generic

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestGridDataKernels(unittest.TestCase):

    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')

        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
        self.image = create_image(npixel=512, cellsize=0.0005, phasecentre=self.phasecentre,
                                  polarisation_frame=PolarisationFrame("stokesI"))
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_fill_wterm_to_list(self):
        gcfcf = create_awterm_convolutionfunction(self.image, make_pb=None, nw=201, wstep=8.0, oversampling=8,
                                                  support=30, use_aaf=True)

        wtowers = convert_kernel_to_list(gcfcf)
        print(wtowers)

    def test_fill_awterm_to_list(self):
        make_pb = functools.partial(create_pb_generic, diameter=35.0, blockage=0.0, use_local=False)
        pb = make_pb(self.image)
        gcfcf = create_awterm_convolutionfunction(self.image, make_pb=make_pb, nw=201, wstep=8, oversampling=8,
                                                  support=30, use_aaf=True)
        wtowers = convert_kernel_to_list(gcfcf)
        print(wtowers)


if __name__ == '__main__':
    unittest.main()
