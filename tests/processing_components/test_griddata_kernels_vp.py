""" Unit tests for image operations


"""
import functools
import logging
import os
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import create_image
from rascil.processing_components.griddata import convert_convolutionfunction_to_image
from rascil.processing_components.griddata.kernels import create_vpterm_convolutionfunction
from rascil.processing_components.image.operations import export_image_to_fits
from rascil.processing_components.imaging.primary_beams import create_vp

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestGridDataKernels(unittest.TestCase):

    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')

        self.phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs',
                                    equinox='J2000')
        self.image = create_image(npixel=512, cellsize=0.0005, phasecentre=self.phasecentre,
                                  polarisation_frame=PolarisationFrame("stokesI"))
        self.persist = os.getenv("RASCIL_PERSIST", False)

    def test_fill_vpterm_to_convolutionfunction(self):
        self.image = create_image(npixel=512, cellsize=0.0005, phasecentre=self.phasecentre,
                                  frequency=numpy.array([1.36e9]),
                                  polarisation_frame=PolarisationFrame("stokesIQUV"))
        make_vp = functools.partial(create_vp, telescope="MID_FEKO_B2")
        gcf, cf = create_vpterm_convolutionfunction(self.image, make_vp=make_vp, oversampling=16,
                                                    support=32, use_aaf=True)
        cf_image = convert_convolutionfunction_to_image(cf)
        cf_image.data = numpy.real(cf_image.data)
        if self.persist:
            export_image_to_fits(cf_image, "%s/test_convolutionfunction_aterm_vp_cf.fits" % self.dir)

        # Tests for the VP convolution function are different because it does not peak
        # at the centre of the uv plane
        peak_location = numpy.unravel_index(numpy.argmax(numpy.abs(cf.data)), cf.shape)
        assert numpy.abs(cf.data[peak_location] - (0.005286010417866583 + 0.000490194742591494j)) < 1e-7, cf.data[
            peak_location]
        assert peak_location == (0, 3, 0, 11, 8, 11, 16), peak_location
        u_peak, v_peak = cf.grid_wcs.sub([1, 2]).wcs_pix2world(peak_location[-2], peak_location[-1], 0)
        assert numpy.abs(u_peak - 19.53125) < 1e-7, u_peak
        assert numpy.abs(v_peak) < 1e-7, u_peak

        if self.persist:
            export_image_to_fits(gcf, "%s/test_convolutionfunction_aterm_vp_gcf.fits" % self.dir)


if __name__ == '__main__':
    unittest.main()
