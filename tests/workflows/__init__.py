"""Offers he base class for rsexecute-based unit tests"""


class rsexecuteTestCase(object):
    """Sets up the rsexecute global object as appropriate, and closes it when done"""
    
    def setUp(self):
        super(rsexecuteTestCase, self).setUp()
        
        import os
        from rascil.wrappers.rsexecute.execution_support import rsexecute
        use_dlg = os.environ.get('ARL_TESTS_USE_DLG', '0') == '1'
        use_dask = os.environ.get('ARL_TESTS_USE_DASK', '1') == '1'
        rsexecute.set_client(use_dask=use_dask, use_dlg=use_dlg)
        
        # Start a daliuge node manager for these tests; make sure it can see
        # the rascil modules. The node manager will be shut down at tearDown
        if use_dlg:
            from dlg import tool
            arl_root = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            self.nm_proc = tool.start_process('nm', ['--dlg-path', arl_root])
    
    def tearDown(self):
        from rascil.wrappers.rsexecute.execution_support import rsexecute
        rsexecute.close()
        if rsexecute.using_dlg:
            from dlg import utils
            utils.terminate_or_kill(self.nm_proc, 10)
        super(rsexecuteTestCase, self).tearDown()
