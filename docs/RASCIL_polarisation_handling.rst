.. _polarisation_handling:

Polarisation handling
*********************

Polarisation handling is intended to implement the Hamaker-Bregman-Sault formalism.

For imaging:

 * Types of polarisation allowed are stokesIQUV, stokesI, linear, circular.
 * These are defined in :py:class:`rascil.data_models.polarisation.PolarisationFrame`
 * Images may be defined as stokesI, stokesIQUV, linear, or circular
 * To convert from Stokes image to polarised image see :py:func:`rascil.processing_components.image.operations.convert_stokes_to_polimage`
 * To convert from polarised image to Stokes image :py:func:`rascil.processing_components.image.operations.convert_polimage_to_stokes`
 * Skycomponents may be defined as stokesI, stokesIQUV, linear, or circular
 * Visibility/BlockVisibility may be defined as stokesI, stokesIQUV, linear, or circular.
 * Dish/station voltage patterns are described by images in which each pixel is a 2 x 2 complex matrix.

For calibration, the Jones matrices allowed are:

 * T = scalar phase-only term i.e. complex unit-amplitude phasor times the identity [2,2] matrix
 * G = vector complex gain i.e. diagonal [2, 2] matrix with different phasors
 * B = Same as G but frequency dependent

