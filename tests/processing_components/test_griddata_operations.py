""" Unit tests for image operations


"""
import logging
import unittest

import numpy

from rascil.processing_components.griddata.operations import create_griddata_from_image, convert_griddata_to_image
from rascil.processing_components.simulation import create_test_image

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestGridData(unittest.TestCase):
    
    def setUp(self):
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        
        self.m31image = create_test_image(cellsize=0.0001)
        self.cellsize = 180.0 * 0.0001 / numpy.pi
    
    def test_create_griddata_from_image(self):
        m31model_by_image = create_griddata_from_image(self.m31image, None)
        assert m31model_by_image.shape[0] == self.m31image.shape[0]
        assert m31model_by_image.shape[1] == self.m31image.shape[1]
        assert m31model_by_image.shape[3] == self.m31image.shape[2]
        assert m31model_by_image.shape[4] == self.m31image.shape[3]
    
    def test_convert_griddata_to_image(self):
        m31model_by_image = create_griddata_from_image(self.m31image, None)
        m31_converted = convert_griddata_to_image(m31model_by_image)
    
if __name__ == '__main__':
    unittest.main()
