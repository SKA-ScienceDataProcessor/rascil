""" Unit tests for testing support


"""
import logging
import unittest

import numpy

# Import the base and then make a global version
from rascil.wrappers.rsexecute.execution_support import rsexecuteBase

log = logging.getLogger(__name__)

class Testrsexecute(unittest.TestCase):
    
    def setUp(self):
        global rsexecute
        rsexecute = rsexecuteBase(use_dask=True)
        rsexecute.set_client(use_dask=True, verbose=False)
        
    def tearDown(self):
        rsexecute.close()
    
    def test_useFunction(self):
        def square(x):
            return x ** 2

        graph = rsexecute.execute(square)(numpy.arange(10))
        result = rsexecute.compute(graph, sync=True)
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all(), result

    def test_useDaskAsync(self):
        def square(x):
            return x ** 2
    
        graph = rsexecute.execute(square)(numpy.arange(10))
        result = rsexecute.compute(graph).result()
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()

    def test_useDaskSync(self):
        def square(x):
            return x ** 2
    
        graph = rsexecute.execute(square)(numpy.arange(10))
        result = rsexecute.compute(graph, sync=True)
        assert (result == numpy.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])).all()
