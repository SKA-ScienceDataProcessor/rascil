.. _rascil_workflows_serial:

.. py:currentmodule:: rascil.workflows.serial


Serial
======

Serial workflows are executed immediately, and should produce the same results as rsexecute workflows.

For example::

    from rascil.workflows import continuum_imaging_list_serial_workflow
    deconvolved_list, residual_list, restored_list = \
        continuum_imaging_list_serial_workflow(vis_list, model_imagelist=model_list,
                                                  context='wstack', vis_slices=51,
                                                  scales=[0, 3, 10], algorithm='mmclean',
                                                  nmoment=3, niter=1000,
                                                  fractional_threshold=0.1, threshold=0.1,
                                                  nmajor=5, gain=0.25,
                                                  psf_support=64)

.. toctree::
   :maxdepth: 1

.. automodapi::    rascil.workflows.serial.calibration
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.serial.imaging
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.serial.pipelines
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.serial.simulation
   :no-inheritance-diagram:

.. automodapi::    rascil.workflows.serial.skymodel
   :no-inheritance-diagram:

