""" Unit tests for voltage pattern operations


"""
import logging
import os
import unittest

import numpy

from rascil.processing_components import copy_image, fft_image, create_vp, qa_image, pad_image
from rascil.processing_components.image.operations import export_image_to_fits

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestImage(unittest.TestCase):
    
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.persist = os.getenv("RASCIL_PERSIST", True)
    
    def test_create_image_(self):
        vp = create_vp(telescope='MID_FEKO_B2')
        pad = 4
        vp_shape = (vp.shape[0], vp.shape[1], pad * vp.shape[2], pad * vp.shape[3])
        vp = pad_image(vp, vp_shape)
        print(qa_image(vp))
        fft_vp = fft_image(vp)
        print(qa_image(fft_vp))
        fft_vp.data = fft_vp.data[:, :, \
        (vp_shape[2] // 2 - vp.shape[2] // pad):(vp_shape[2] // 2 + vp.shape[2] // pad),
        (vp_shape[2] // 2 - vp.shape[2] // pad):(vp_shape[2] // 2 + vp.shape[2] // pad)]
        
        real_fft_vp = copy_image(fft_vp)
        real_fft_vp.data = real_fft_vp.data.real
        print(qa_image(real_fft_vp))
        imag_fft_vp = copy_image(fft_vp)
        imag_fft_vp.data = imag_fft_vp.data.imag
        print(qa_image(imag_fft_vp))
        
        if self.persist:
            export_image_to_fits(real_fft_vp, fitsfile='{}/test_vp_real.fits'.format(self.dir))
            export_image_to_fits(imag_fft_vp, fitsfile='{}/test_vp_imag.fits'.format(self.dir))
        
        ifft_fft_vp = fft_image(fft_vp, template_image=vp)
        
        assert numpy.max(numpy.abs(vp.data - ifft_fft_vp.data)) < 1e-12


if __name__ == '__main__':
    unittest.main()
