""" Functions for imaging from visibility data.

The functions include 2D prediction and inversion operations. A very simple example, given a model Image to specify the image size, sampling, and phasecentre::

    model = create_image_from_visibility(vis, npixel=1024, nchan=1)
    dirty, sumwt = invert_2d(vis, model)

The call to create_image_from_visibility step constructs a template image. The dirty image is constructed according to this template.

AW projection is supported by the predict_2d and invert_2d methods, provided the gridding kernel is constructed and passed in. For example::

    gcfcf = create_awterm_convolutionfunction(model, nw=100, wstep=8.0, oversampling=8,
        support=100, use_aaf=True)
    dirty, sumwt = invert_2d(vis, model, gcfcf = gcfcf)

If installed, the nifty gridder (https://gitlab.mpcdf.mpg.de/ift/nifty_gridder) can also be used::

    dirty, sumwt = invert_ng(vis, model, verbosity=2)

These functions can be used directly. For distribution, these functions can be orchestrated by the rsexecute/Dask framework. This allows w stacking, timeslicing, and a wprojection/w stacking hybrid. See

    :py:mod:`rascil.workflows.rsexecute.imaging`
    :py:mod:`rascil.workflows.serial.imaging`

The convolutional gridding functions are to be found in griddata module

    :py:mod:`rascil.processing_components.griddata`

"""
from .base import *
from .primary_beams import *
from .imaging_params import *
from .timeslice_single import *
from .weighting import *
from .wstack_single import *
from .dft import *

