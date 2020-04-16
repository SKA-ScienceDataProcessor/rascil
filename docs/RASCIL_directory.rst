
.. toctree::
   :maxdepth: 2

Directory
*********

This directory is designed to to help those familiar with other calibration and imaging packages navigate the Algorithm
Reference Library. Not all functions are listed here. The API contains all functions.

The long form of the name is given for all entries but all function names arre unique so a given function can be
accessed using the very top level import::

   import rascil.data_models
   import rascil.processing_components
   import rascil.workflows


Data containers used by RASCIL
==============================

RASCIL holds data in python Classes. The bulk data is usually kept in a python structured array, and the meta data as
attributes.

See :py:mod:`rascil.data_models.memory_data_models` for the following definitions:

* Image (data and WCS header): :py:class:`rascil.data_models.memory_data_models.Image`
* Skycomponent (data for a point source or a Gaussian source): :py:class:`rascil.data_models.memory_data_models.Skycomponent`
* SkyModel (collection of SkyComponents and Images): :py:class:`rascil.data_models.memory_data_models.SkyModel`
* Antenna-based visibility table, shape (nants, nants, nchan, npol), length ntimes): :py:class:`rascil.data_models.memory_data_models.BlockVisibility`
* Baseline based visibility tables shape (npol,), length nvis) :py:class:`rascil.data_models.memory_data_models.Visibility`
* Telescope Configuration: :py:class:`rascil.data_models.memory_data_models.Configuration`
* GainTable for gain solutions (as (e.g. output from solve_gaintable): :py:class:`rascil.data_models.memory_data_models.GainTable`

Functions
=========

Create empty visibility data set for observation
------------------------------------------------

* For Visibility: :py:func:`rascil.processing_components.visibility.base.create_visibility`
* For BlockVisibility: :py:func:`rascil.processing_components.visibility.base.create_blockvisibility`

Read existing Measurement Set
-----------------------------

Casacore must be installed for MS reading and writing:

* List contents of a MeasurementSet: :py:func:`rascil.processing_components.visibility.base.list_ms`
* Creates a list of Visibilities, one per FIELD_ID and DATA_DESC_ID: :py:func:`rascil.processing_components.visibility.base.create_visibility_from_ms`
* Creates a list of BlockVisibilities, one per FIELD_ID and DATA_DESC_ID: :py:func:`rascil.processing_components.visibility.base.create_blockvisibility_from_ms`

Visibility gridding and degridding
----------------------------------

* Convolutional gridding: :py:func:`rascil.processing_components.griddata.gridding.grid_visibility_to_griddata`
* Convolutional degridding: :py:func:`rascil.processing_components.griddata.gridding.degrid_visibility_from_griddata`

Visibility weighting and tapering
---------------------------------

* Weighting: :py:func:`rascil.processing_components.imaging.weighting.weight_visibility`
* Gaussian tapering: :py:func:`rascil.processing_components.imaging.weighting.taper_visibility_gaussian`
* Tukey tapering: :py:func:`rascil.processing_components.imaging.weighting.taper_visibility_tukey`

Visibility predict and invert
-----------------------------

* Predict BlockVisibility or Visibility for Skycomponent :py:func:`rascil.processing_components.imaging.base.predict_skycomponent_visibility`
* Predict by de-gridding visibilities :py:func:`rascil.processing_components.imaging.base.predict_2d`
* Invert by gridding visibilities :py:func:`rascil.processing_components.imaging.base.invert_2d`
* Predict by de-gridding visibilities with Nifty Gridder :py:func:`rascil.processing_components.imaging.ng.predict_ng`
* Invert by gridding visibilities with Nifty Gridder :py:func:`rascil.processing_components.imaging.ng.invert_ng`

Deconvolution
-------------

* Deconvolution :py:func:`rascil.processing_components.image.deconvolution.deconvolve_cube` wraps:

 * Hogbom Clean: :py:func:`rascil.processing_components.arrays.cleaners.hogbom`
 * Hogbom Complex Clean: :py:func:`rascil.processing_components.arrays.cleaners.hogbom_complex`
 * Multi-scale Clean: :py:func:`rascil.processing_components.arrays.cleaners.msclean`
 * Multi-scale multi-frequency Clean: :py:func:`rascil.processing_components.arrays.cleaners.msmfsclean`


* Restore: :py:func:`rascil.processing_components.image.deconvolution.restore_cube`

Calibration
-----------

* Create empty gain table: :py:func:`rascil.processing_components.calibration.operations.create_gaintable_from_blockvisibility`
* Solve for complex gains: :py:func:`rascil.processing_components.calibration.solvers.solve_gaintable`
* Apply complex gains: :py:func:`rascil.processing_components.calibration.operations.apply_gaintable`

Coordinate transforms
---------------------

* Phase rotation: :py:func:`rascil.processing_components.visibility.base.phaserotate_visibility`
* Station/baseline (XYZ <-> UVW): :py:mod:`rascil.processing_components.util.coordinate_support`
* Source (spherical -> tangent plane): :py:mod:`rascil.processing_components.util.coordinate_support`

Image
-----

* Image operations: :py:func:`rascil.processing_components.image.operations`
* Import from FITS: :py:func:`rascil.processing_components.image.operations.import_image_from_fits`
* Export from FITS: :py:func:`rascil.processing_components.image.operations.export_image_to_fits`
* Reproject coordinate system: :py:func:`rascil.processing_components.image.operations.reproject_image`
* Smooth image: :py:func:`rascil.processing_components.image.operations.smooth_image`
* FFT: :py:func:`rascil.processing_components.image.operations.fft_image`
* Remove continuum: :py:func:`rascil.processing_components.image.operations.remove_continuum_image`
* Convert polarisation:

 * From Stokes To Polarisation: :py:func:`rascil.processing_components.image.operations.convert_stokes_to_polimage`
 * From Polarisation to Stokes: :py:func:`rascil.processing_components.image.operations.convert_polimage_to_stokes`


Visibility
----------

* Append/sum/divide/QA: :py:func:`rascil.processing_components.visibility.operations.divide_visibility`
* Remove continuum: :py:func:`rascil.processing_components.visibility.operations.remove_continuum_blockvisibility`
* Integrate across channels: :py:func:`rascil.processing_components.visibility.operations.integrate_visibility_by_channel`
* Coalesce (i.e. BDA) :py:func:`rascil.processing_components.visibility.coalesce.coalesce_visibility`
* Decoalesce (i.e. BDA) :py:func:`rascil.processing_components.visibility.coalesce.decoalesce_visibility`

Workflows
=========

Workflows coordinate processing using the data models, processing components, and processing library. These are high
level functions, and are available in rsexecute (i.e. dask) version and sometimes scalar version.

Calibration workflows
---------------------

* Calibrate workflow: :py:func:`rascil.workflows.rsexecute.calibration.calibrate_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.calibration.calibrate_list_serial_workflow`


Imaging workflows
-----------------

* Invert: :py:func:`rascil.workflows.rsexecute.imaging.invert_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.imaging.invert_list_serial_workflow`
* Predict: :py:func:`rascil.workflows.rsexecute.imaging.predict_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.imaging.predict_list_serial_workflow`
* Deconvolve: :py:func:`rascil.workflows.rsexecute.imaging.deconvolve_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.imaging.deconvolve_list_serial_workflow`

Pipeline workflows
------------------

* ICAL: :py:func:`rascil.workflows.rsexecute.pipelines.ical_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.pipelines.ical_list_serial_workflow`
* Continuum imaging: :py:func:`rascil.workflows.rsexecute.pipelines.continuum_imaging_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.pipelines.continuum_imaging_list_serial_workflow`
* Spectral line imaging: :py:func:`rascil.workflows.rsexecute.pipelines.spectral_line_imaging_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.pipelines.spectral_line_imaging_list_serial_workflow`
* MPCCAL: :py:func:`rascil.workflows.rsexecute.pipelines.mpccal_skymodel_list_rsexecute_workflow`

Simulation workflows
--------------------

* Testing and simulation support: :py:func:`rascil.workflows.rsexecute.simulation.simulate_list_rsexecute_workflow` :py:func:`rascil.workflows.serial.simulation.simulate_list_serial_workflow`

Execution
---------

* Execution framework (an interface to Dask): :py:func:`rascil.workflows.rsexecute.execution_support`



