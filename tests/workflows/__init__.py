"""Offers he base class for rsexecute-based unit tests"""


class rsexecuteTestCase(object):
    """Sets up the rsexecute global object as appropriate, and closes it when done"""
    
    def setUp(self):
#        super(rsexecuteTestCase, self).setUp()
        
        import os
        from rascil.wrappers.rsexecute.execution_support import rsexecute
        use_dask = os.environ.get('ARL_TESTS_USE_DASK', '1') == '1'
        rsexecute.set_client(use_dask=use_dask)
        
    def tearDown(self):
        from rascil.wrappers.rsexecute.execution_support import rsexecute
        rsexecute.close()
