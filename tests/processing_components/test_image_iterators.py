"""Unit tests for image iteration


"""
import logging
import unittest

import numpy

from rascil.data_models.polarisation import PolarisationFrame

from rascil.processing_components.image.iterators import image_raster_iter, image_channel_iter, \
    image_null_iter

from rascil.processing_components.image.operations import create_empty_image_like, pad_image
from rascil.processing_components.simulation import create_test_image

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestImageIterators(unittest.TestCase):
    
    def get_test_image(self):
        testim = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
        return pad_image(testim, [1, 1, 512, 512])
    
    def test_raster(self):
        
        m31original = self.get_test_image()
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
        
        for nraster in [1, 2, 4, 8, 9]:
            m31model = self.get_test_image()
            for patch in image_raster_iter(m31model, facets=nraster):
                assert patch.data.shape[3] == (m31model.data.shape[3] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[3],
                                                                                (m31model.data.shape[3] // nraster))
                assert patch.data.shape[2] == (m31model.data.shape[2] // nraster), \
                    "Number of pixels in each patch: %d not as expected: %d" % (patch.data.shape[2],
                                                                                (m31model.data.shape[2] // nraster))
                patch.data *= 2.0
            
            diff = m31model.data - 2.0 * m31original.data
            assert numpy.max(numpy.abs(m31model.data)), "Raster is empty for %d" % nraster
            assert numpy.max(numpy.abs(diff)) == 0.0, "Raster set failed for %d" % nraster

    def test_raster_exception(self):
    
        m31original = self.get_test_image()
        assert numpy.max(numpy.abs(m31original.data)), "Original is empty"
    
        for nraster, overlap in [(-1, -1), (-1, 0), (2, 128), (1e6, 127)]:
            
            with self.assertRaises(AssertionError):
                m31model = create_test_image(polarisation_frame=PolarisationFrame('stokesI'))
                for patch in image_raster_iter(m31model, facets=nraster, overlap=overlap):
                    patch.data *= 2.0
            
    def test_channelise(self):
        m31cube = create_test_image(polarisation_frame=PolarisationFrame('stokesI'),
                                        frequency=numpy.linspace(1e8,1.1e8, 128))
        
        for subimages in [128, 16, 8, 2, 1]:
            for slab in image_channel_iter(m31cube, subimages=subimages):
                assert slab.data.shape[0] == 128 // subimages

    def test_null(self):
        m31cube = create_test_image(polarisation_frame=PolarisationFrame('stokesI'),
                                    frequency=numpy.linspace(1e8, 1.1e8, 128))
    
        for i, im in enumerate(image_null_iter(m31cube)):
            assert i<1, "Null iterator returns more than one value"


if __name__ == '__main__':
    unittest.main()
