""" Execute wrap dask such that with the same code Dask.delayed can be replaced by immediate calculation

"""

__all__ = ['rsexecuteBase', 'rsexecute']

from rascil.wrappers.rsexecute.execution_support.rsexecutebase import rsexecuteBase
rsexecute = rsexecuteBase(use_dask=True)