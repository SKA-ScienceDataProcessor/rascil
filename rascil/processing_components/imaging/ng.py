"""
Functions that implement prediction of and imaging from visibilities using the nifty gridder.

https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

This performs all necessary w term corrections, to high precision.

"""

__all__ = ['predict_ng', 'invert_ng']

import logging
from typing import Union

import numpy

from rascil.data_models.memory_data_models import BlockVisibility, \
    Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import convert_pol_frame
from rascil.processing_components.image.operations import copy_image, \
    image_is_canonical
from rascil.processing_components.imaging.base import shift_vis_to_image, \
    normalize_sumwt, fill_vis_for_psf
from rascil.processing_components.visibility.base import copy_visibility


log = logging.getLogger('logger')

try:
    import nifty_gridder as ng
    
    
    def predict_ng(bvis: BlockVisibility, model: Image, **kwargs) -> BlockVisibility:
        """ Predict using convolutional degridding.
        
        Nifty-gridder version. https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
    
        In the imaging and pipeline workflows, this may be invoked using context='ng'.

        :param bvis: BlockVisibility to be predicted
        :param model: model image
        :return: resulting BlockVisibility (in place works)
        """
        
        assert isinstance(bvis, BlockVisibility), bvis
        assert image_is_canonical(model)
        
        if model is None:
            return bvis
        
        nthreads = get_parameter(kwargs, "threads", 4)
        epsilon = get_parameter(kwargs, "epsilon", 1e-12)
        do_wstacking = get_parameter(kwargs, "do_wstacking", True)
        verbosity = get_parameter(kwargs, "verbosity", 0)
        
        newbvis = copy_visibility(bvis, zero=True)
        
        # Extracting data from BlockVisibility
        freq = bvis.frequency  # frequency, Hz
        nrows, nants, _, vnchan, vnpol = bvis.vis.shape
        
        uvw = newbvis.data['uvw'].reshape([nrows * nants * nants, 3])
        vist = numpy.zeros([vnpol, vnchan, nants * nants * nrows], dtype='complex')
        
        # Get the image properties
        m_nchan, m_npol, ny, nx = model.data.shape
        # Check if the number of frequency channels matches in bvis and a model
        #        assert (m_nchan == v_nchan)
        assert (m_npol == vnpol)
        
        fuvw = uvw.copy()
        # We need to flip the u and w axes. The flip in w is equivalent to the conjugation of the
        # convolution function grid_visibility to griddata
        fuvw[:, 0] *= -1.0
        fuvw[:, 2] *= -1.0
        
        # Find out the image size/resolution
        pixsize = numpy.abs(numpy.radians(model.wcs.wcs.cdelt[0]))
        
        # Make de-gridding over a frequency range and pol fields
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        
        mfs = m_nchan == 1

        if mfs:
            for vpol in range(vnpol):
                vist[vpol, : , :] = ng.dirty2ms(fuvw.astype(numpy.float64),
                                               bvis.frequency.astype(numpy.float64),
                                               model.data[0, vpol, :, :].T.astype(numpy.float64),
                                               pixsize_x=pixsize,
                                               pixsize_y=pixsize,
                                               epsilon=epsilon,
                                               do_wstacking=do_wstacking,
                                               nthreads=nthreads,
                                               verbosity=verbosity).T

        else:
            for vpol in range(vnpol):
                for vchan in range(vnchan):
                    imchan = vis_to_im[vchan]
                    vist[vpol, vchan, :] = ng.dirty2ms(fuvw.astype(numpy.float64),
                                                       numpy.array(freq[vchan:vchan + 1]).astype(numpy.float64),
                                                       model.data[imchan, vpol, :, :].T.astype(numpy.float64),
                                                       pixsize_x=pixsize,
                                                       pixsize_y=pixsize,
                                                       epsilon=epsilon,
                                                       do_wstacking=do_wstacking,
                                                       nthreads=nthreads,
                                                       verbosity=verbosity)[:, 0]
        
        vis = convert_pol_frame(vist.T, model.polarisation_frame, bvis.polarisation_frame, polaxis=2)

        newbvis.data['vis'] = vis.reshape([nrows, nants, nants, vnchan, vnpol])
    
        # Now we can shift the visibility from the image frame to the original visibility frame
        return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)

    
    def invert_ng(bvis: BlockVisibility, model: Image, dopsf: bool = False,
                  normalize: bool = True,
                  **kwargs) -> (Image, numpy.ndarray):
        """ Invert using nifty-gridder module
        
        https://gitlab.mpcdf.mpg.de/ift/nifty_gridder
    
        Use the image im as a template. Do PSF in a separate call.

        In the imaging and pipeline workflows, this may be invoked using context='ng'.

        :param dopsf: Make the PSF instead of the dirty image
        :param bvis: BlockVisibility to be inverted
        :param im: image template (not changed)
        :param normalize: Normalize by the sum of weights (True)
        :return: (resulting image, sum of the weights for each frequency and polarization)
    
        """
        assert image_is_canonical(model)
        
        assert isinstance(bvis, BlockVisibility), bvis
        
        im = copy_image(model)
        
        nthreads = get_parameter(kwargs, "threads", 4)
        epsilon = get_parameter(kwargs, "epsilon", 1e-12)
        do_wstacking = get_parameter(kwargs, "do_wstacking", True)
        verbosity = get_parameter(kwargs, "verbosity", 0)
        
        sbvis = copy_visibility(bvis)
        sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)
        
        freq = sbvis.frequency  # frequency, Hz
        
        nrows, nants, _, vnchan, vnpol = sbvis.vis.shape
        # if dopsf:
        #     sbvis = fill_vis_for_psf(sbvis)

        ms = sbvis.vis.reshape([nrows * nants * nants, vnchan, vnpol])
        ms = convert_pol_frame(ms, bvis.polarisation_frame, im.polarisation_frame, polaxis=2)

        uvw = sbvis.uvw.reshape([nrows * nants * nants, 3])
        wgt = sbvis.flagged_imaging_weight.reshape([nrows * nants * nants, vnchan, vnpol])

        if epsilon > 5.0e-6:
            ms = ms.astype("c8")
            wgt = wgt.astype("f4")
        
        # Find out the image size/resolution
        npixdirty = im.nwidth
        pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
        
        fuvw = uvw.copy()
        # We need to flip the u and w axes.
        fuvw[:, 0] *= -1.0
        fuvw[:, 2] *= -1.0
        
        nchan, npol, ny, nx = im.shape
        im.data[...] = 0.0
        sumwt = numpy.zeros([nchan, npol])
        
        # There's a latent problem here with the weights.
        # wgt = numpy.real(convert_pol_frame(wgt, bvis.polarisation_frame, im.polarisation_frame, polaxis=2))
        
        # Set up the conversion from visibility channels to image channels
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        
       # Nifty gridder likes to receive contiguous arrays so we transpose
        # at the beginning
        
        mfs = nchan == 1
        if dopsf:
            
            mst = ms.T
            mst[...] = 0.0
            mst[0, ...] = 1.0
            wgtt = wgt.T

            if mfs:
                dirty = ng.ms2dirty(fuvw.astype(numpy.float64),
                                    bvis.frequency.astype(numpy.float64),
                                    numpy.ascontiguousarray(mst[0, :, :].T),
                                    numpy.ascontiguousarray(wgtt[0, :, :].T),
                                    npixdirty, npixdirty, pixsize, pixsize, epsilon,
                                    do_wstacking=do_wstacking,
                                    nthreads=nthreads, verbosity=verbosity)
                sumwt[0, :] += numpy.sum(wgtt[0, 0, :].T, axis=0)
                im.data[0, :] += dirty.T
            else:
                for vchan in range(vnchan):
                    ichan = vis_to_im[vchan]
                    frequency = numpy.array(freq[vchan:vchan + 1]).astype(numpy.float64)
                    dirty = ng.ms2dirty(fuvw.astype(numpy.float64),
                                        frequency.astype(numpy.float64),
                                        numpy.ascontiguousarray(mst[0, vchan, :][..., numpy.newaxis]),
                                        numpy.ascontiguousarray(wgtt[0, vchan, :][..., numpy.newaxis]),
                                        npixdirty, npixdirty, pixsize, pixsize, epsilon,
                                        do_wstacking=do_wstacking,
                                        nthreads=nthreads, verbosity=verbosity)
                    sumwt[ichan, :] += numpy.sum(wgtt[0, ichan, :].T, axis=0)
                    im.data[ichan, :] += dirty.T
        else:
            mst = ms.T
            wgtt = wgt.T
            for pol in range(npol):
                if mfs:
                    dirty = ng.ms2dirty(fuvw.astype(numpy.float64),
                                        bvis.frequency.astype(numpy.float64),
                                        numpy.ascontiguousarray(mst[pol, :, :].T),
                                        numpy.ascontiguousarray(wgtt[pol, :, :].T),
                                        npixdirty, npixdirty, pixsize, pixsize, epsilon,
                                        do_wstacking=do_wstacking,
                                        nthreads=nthreads, verbosity=verbosity)
                    sumwt[0, pol] += numpy.sum(wgtt[pol, 0, :].T, axis=0)
                    im.data[0, pol] += dirty.T
                else:
                    for vchan in range(vnchan):
                        ichan = vis_to_im[vchan]
                        frequency = numpy.array(freq[vchan:vchan + 1]).astype(numpy.float64)
                        dirty = ng.ms2dirty(fuvw.astype(numpy.float64),
                                            frequency.astype(numpy.float64),
                                            numpy.ascontiguousarray(mst[pol, vchan, :][..., numpy.newaxis]),
                                            numpy.ascontiguousarray(wgtt[pol, vchan, :][..., numpy.newaxis]),
                                            npixdirty, npixdirty, pixsize, pixsize, epsilon,
                                            do_wstacking=do_wstacking,
                                            nthreads=nthreads, verbosity=verbosity)
                        sumwt[ichan, pol] += numpy.sum(wgtt[pol, ichan, :].T, axis=0)
                        im.data[ichan, pol] += dirty.T

        
        if normalize:
            im = normalize_sumwt(im, sumwt)
        
        return im, sumwt

except ImportError:
    import warnings
    
    warnings.warn('Cannot import nifty_gridder, ng disabled', ImportWarning)
    
    
    def predict_ng(bvis: BlockVisibility, model: Image, **kwargs) -> BlockVisibility:
        """ Predict using convolutional degridding.

        Nifty-gridder version. https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

        In the imaging and pipeline workflows, this may be invoked using context='ng'.

        :param bvis: BlockVisibility to be predicted
        :param model: model image
        :return: resulting BlockVisibility (in place works)
        """

        log.error("Nifty gridder not available")
        return bvis
    
    
    def invert_ng(bvis: BlockVisibility, model: Image, dopsf: bool = False,
                  normalize: bool = True,
                  **kwargs) -> (Image, numpy.ndarray):
        """ Invert using nifty-gridder module

        https://gitlab.mpcdf.mpg.de/ift/nifty_gridder

        Use the image im as a template. Do PSF in a separate call.

        Any shifting needed is performed here.

        In the imaging and pipeline workflows, this may be invoked using context='ng'.

        :param bvis: BlockVisibility to be inverted
        :param im: image template (not changed)
        :param normalize: Normalize by the sum of weights (True)
        :return: (resulting image, sum of the weights for each frequency and polarization)

        """
        log.error("Nifty gridder not available")
        return model, None
