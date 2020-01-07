.. _rascil_processing_components_calibration:

.. py:currentmodule:: rascil.processing_components.calibration

***********
Calibration
***********

Calibration is performed by fitting observed visibilities to a model visibility.

The scalar equation to be minimised is:

.. math:: S = \sum_{t,f}^{}{\sum_{i,j}^{}{w_{t,f,i,j}\left| V_{t,f,i,j}^{\text{obs}} - J_{i}{J_{j}^{*}V}_{t,f,i,j}^{\text{mod}} \right|}^{2}}

The least squares fit algorithm uses an iterative substitution (or relaxation) algorithm from Larry D'Addario in the
late seventies.

.. toctree::
   :maxdepth: 3

.. automodapi::    rascil.processing_components.calibration.chain_calibration
   :no-inheritance-diagram:

.. automodapi::    rascil.processing_components.calibration.iterators
   :no-inheritance-diagram:

.. automodapi::    rascil.processing_components.calibration.operations
   :no-inheritance-diagram:

.. automodapi::    rascil.processing_components.calibration.pointing
   :no-inheritance-diagram:

.. automodapi::    rascil.processing_components.calibration.rcal
   :no-inheritance-diagram:

.. automodapi::    rascil.processing_components.calibration.solvers
   :no-inheritance-diagram:


