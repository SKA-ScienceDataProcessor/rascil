""" Unit processing_components for pipelines


"""

import unittest

from rascil.data_models.parameters import get_parameter

import logging

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestParameters(unittest.TestCase):
    def setUp(self):
        self.parameters = {'npixel': 256, 'cellsize': 0.1, 'spectral_mode': 'mfs'}

    def test_getparameter(self):
    
        def t1(**kwargs):
            assert get_parameter(kwargs, 'cellsize') == 0.1
            assert get_parameter(kwargs, 'spectral_mode', 'channels') == 'mfs'
            assert get_parameter(kwargs, 'null_mode', 'mfs') == 'mfs'
            assert get_parameter(kwargs, 'foo', 'bar') == 'bar'
            assert get_parameter(kwargs, 'foo') is None
            assert get_parameter(None, 'foo', 'bar') == 'bar'
        
        kwargs = self.parameters
        t1(**kwargs)


if __name__ == '__main__':
    unittest.main()
