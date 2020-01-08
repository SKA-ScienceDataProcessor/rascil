.. _rascil_workflows:

.. py:currentmodule:: rascil.workflows

Workflows
*********

Workflows are higher level functions that make use of the processing components, and processing library, operating on data
models. They are available in serial and rsexecute versions.

rsexecute workflows can be used in two modes

 - delayed using `Dask.delayed <https://docs.dask.org/en/latest/delayed.html>`_
 - serially executed immediately on definition,

Distribution is acheived by working on lists of data models, such as lists of BlockVisibilities.

For example::

    from rascil.workflows import continuum_imaging_list_rsexecute_workflow, rsexecute
    rsexecute.set_client(use_dask=True, threads_per_worker=1,
        memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
        local_dir=dask_dir, verbose=True)
    continuum_imaging_list = \
        continuum_imaging_list_rsexecute_workflow(vis_list, model_imagelist=model_list,
                                                  context='wstack', vis_slices=51,
                                                  scales=[0, 3, 10], algorithm='mmclean',
                                                  nmoment=3, niter=1000,
                                                  fractional_threshold=0.1, threshold=0.1,
                                                  nmajor=5, gain=0.25,
                                                  psf_support=64)

    deconvolved_list, residual_list, restored_list = \
        rsexecute.compute(continuum_imaging_list, sync=True)

The call to continuum_imaging_list_rsexecute_workflow does not execute immediately just generates a
Dask.delayed object that can be computed subsequently. The higher level functions such as
continuum_imaging_list_rsexecute_workflow are built from lower level functions such as
invert_list_rsexecute_workflow.

In this example, changing use_dask to False will cause the definitions to be executed immediately. Alternatively, the
serial version could be used::

    from rascil.workflows import continuum_imaging_list_serial_workflow
    deconvolved_list, residual_list, restored_list = \
        continuum_imaging_list_serial_workflow(vis_list, model_imagelist=model_list,
                                               context='wstack', vis_slices=51,
                                               scales=[0, 3, 10], algorithm='mmclean',
                                               nmoment=3, niter=1000,
                                               fractional_threshold=0.1, threshold=0.1,
                                               nmajor=5, gain=0.25,
                                               psf_support=64)

Most workflows are available in both serial and rsexecute versions, recognising that the optimisations for the
two cases are different.

.. toctree::
   :maxdepth: 1

   rsexecute/index
   serial/index
   shared/index


