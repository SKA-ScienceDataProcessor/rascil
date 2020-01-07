""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid after using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the griddata
convolution function by division in the image plane of the transform.

This approach may be extended to include image plane effect such as the w term and the antenna/station primary beam.

This module contains functions for performing the griddata process and the inverse degridding process.

"""
from .convolution_functions import *
from .gridding import *
from .kernels import *
from .operations import *
