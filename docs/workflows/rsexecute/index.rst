.. _rascil_workflows_rsexecute:

.. py:currentmodule:: rascil.workflows.rsexecute

rsexecute
=========

rsexecute workflows can be used in two modes

 - delayed using `Dask.delayed <https://docs.dask.org/en/latest/delayed.html>`_
 - serially executed immediately on definition,

Distribution is acheived by working on lists of data models, such as lists of BlockVisibilities.

The rsexecute framework relies upon a singleton object called rsexecute. This is documented below
as the class _rsexecutebase.

For example::

    from rascil.workflows import continuum_imaging_list_rsexecute_workflow, rsexecute
    rsexecute.set_client(use_dask=True, threads_per_worker=1,
        memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
        local_dir=dask_dir, verbose=True)
    continuum_imaging_list = continuum_imaging_list_rsexecute_workflow(vis_list,
        model_imagelist=model_list,
        context='wstack', vis_slices=51,
        scales=[0, 3, 10], algorithm='mmclean',
        nmoment=3, niter=1000,
        fractional_threshold=0.1, threshold=0.1,
        nmajor=5, gain=0.25,
        psf_support=64)

    deconvolved_list, residual_list, restored_list = rsexecute.compute(continuum_imaging_list,
        sync=True)

The call to continuum_imaging_list_rsexecute_workflow does not execute immediately just generates a
Dask.delayed object that can be computed subsequently. The higher level functions such as
continuum_imaging_list_rsexecute_workflow are built from lower level functions such as
invert_list_rsexecute_workflow.

In this example, changing use_dask to False will cause the definitions to be executed immediately. Alternatively, the
serial version could be used::

    from rascil.workflows import continuum_imaging_list_serial_workflow
    deconvolved_list, residual_list, restored_list =
    continuum_imaging_list = continuum_imaging_list_serial_workflow(vis_list,
        model_imagelist=model_list,
        context='wstack', vis_slices=51,
        scales=[0, 3, 10], algorithm='mmclean',
        nmoment=3, niter=1000,
        fractional_threshold=0.1, threshold=0.1,
        nmajor=5, gain=0.25,
        psf_support=64)


Most workflows are available in both serial and rsexecute versions, recognising that the optimisations for the
two cases are different.


.. toctree::
   :maxdepth: 2

.. automodapi::    rascil.workflows.rsexecute.calibration
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.rsexecute.image
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.rsexecute.imaging
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.rsexecute.pipelines
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.rsexecute.simulation
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.rsexecute.skymodel
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.rsexecute.execution_support
   :no-inheritance-diagram:

Classes
%%%%%%%

The rsexecute framework relies upon a singleton object called rsexecute. This is documented below
as the class _rsexecutebase. Note that by design it is not possible to create more than
one _rsexecutebase object.


.. autoclass::    rascil.workflows.rsexecute.execution_support.rsexecute._rsexecutebase
   :members:
