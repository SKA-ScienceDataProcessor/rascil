""" Execute wrap dask such that with the same code Dask.delayed can be replaced by immediate calculation

"""

__all__ = ['ARLExecuteBase', 'arlexecute']

from rascil.wrappers.arlexecute.execution_support.arlexecutebase import ARLExecuteBase
arlexecute = ARLExecuteBase(use_dask=True)