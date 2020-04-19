"""
Functions that implement prediction of and imaging from visibilities using the simple, w-projection and w-towers gridder.

https://gitlab.com/ska-telescope/py-wtowers

"""

__all__ = ['vis2dirty', 'gcf2wkern', 'predict_wt', 'invert_wt']

import logging
from typing import Union

import numpy

from rascil.data_models.memory_data_models import Visibility, BlockVisibility, Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import convert_pol_frame
from rascil.processing_components.griddata import convert_kernel_to_list
from rascil.processing_components.image.operations import copy_image, image_is_canonical, export_image_to_fits
from rascil.processing_components.imaging.base import shift_vis_to_image, normalize_sumwt, create_image_from_array
from rascil.processing_components.visibility.base import copy_visibility
from rascil.processing_components.fourier_transforms import ifft, fft

log = logging.getLogger(__name__)

try:
    import wtowers.wtowers as wtowers
    
    def vis2dirty(grid_size, theta, wtvis, wtkern=None, subgrid_size=None, margin=None, winc=None):
        uvgrid = numpy.zeros(grid_size * grid_size, dtype=numpy.complex128)
        if wtkern is None:
            # Simple gridding
            flops = wtowers.grid_simple_func(uvgrid, grid_size, theta, wtvis)
        else:
            if subgrid_size == None or margin == None or winc == None:
                # W-projection gridding
                flops = wtowers.grid_wprojection_func(uvgrid, grid_size, theta, wtvis, wtkern)
            else:
                # W-towers gridding
                flops = wtowers.grid_wtowers_func(uvgrid, grid_size, theta, wtvis, wtkern, subgrid_size, margin, winc)
        # Fill a hermitian conjugated part of the uv_grid plane
        wtowers.make_hermitian_func(uvgrid, grid_size)
        # Create a dirty image and show
        uvgrid = uvgrid.reshape((grid_size, grid_size))
        img = ifft(uvgrid) * float(grid_size)**2
        return numpy.real(img)
    
    
    def dirty2vis(dirty, grid_size, theta, wtvis, wtkern=None, subgrid_size=None, margin=None, winc=None):
        # 1. Make uvgrid
        uvgrid = fft(dirty)
        # 2. Make degridding
        if wtkern is None:
            # Simple degridding
            flops = wtowers.degrid_simple_func(wtvis, uvgrid.reshape((grid_size * grid_size)), grid_size, theta)
        else:
            if subgrid_size == None or margin == None or winc == None:
                # W-projection degridding
                flops = wtowers.degrid_wprojection_func(wtvis, uvgrid.reshape((grid_size * grid_size)), grid_size,
                                                        theta, wtkern)
            else:
                # W-towers degridding
                flops = wtowers.degrid_wtowers_func(wtvis, uvgrid.reshape((grid_size * grid_size)), grid_size, theta,
                                                    wtkern, subgrid_size, margin, winc)
        # 3. Return wtvis
        return wtvis
    
    
    def gcf2wkern(gcfcf, wtkern, im=None, conjugate=False):
        # Get data and metadata from the input gcfcf
        plane_count, wplanes, wmin, wmax, wstep, size_y, size_x, oversampling = convert_kernel_to_list(gcfcf)
        # Allocate memory for wtkern structure
        status = wtowers.wkernel_allocate_func(wtkern, plane_count, size_x, size_y, oversampling)
        # Copy the rest of the metadata to wtkern
        wtkern.w_min = wmin[0]
        wtkern.w_max = wmax[0]
        wtkern.w_step = wstep
        wtkern.oversampling = oversampling
        
        # Copy w-kernels into wtkern.kern_by_w
        for i in range(plane_count):
            w = wplanes[i][0][0]
            wtkern.kern_by_w[i].w = wplanes[i][0][0]
            wplanes_f = numpy.asarray(wplanes[i][1])

            wplanes_f = wplanes_f.flatten()

            if conjugate:
                for idx in range(wplanes_f.shape[0]):
                    wtkern.kern_by_w[i].data[2 * idx] = numpy.real(wplanes_f[idx])
                    wtkern.kern_by_w[i].data[2 * idx + 1] = - numpy.imag(wplanes_f[idx])
            else:
                for idx in range(wplanes_f.shape[0]):
                    wtkern.kern_by_w[i].data[2 * idx] = numpy.real(wplanes_f[idx])
                    wtkern.kern_by_w[i].data[2 * idx + 1] = numpy.imag(wplanes_f[idx])

        return wtkern

    def predict_wt(bvis: BlockVisibility, model: Image, gcfcf=None, **kwargs) -> \
            BlockVisibility:
        """ Predict using convolutional degridding.
        
        Wtowers version. https://gitlab.com/ska-telescope/py-wtowers
    
        :param gcfcf:
        :param bvis: BlockVisibility to be predicted
        :param model: model image
        :return: resulting BlockVisibility (in place works)
        """
        
        assert isinstance(bvis, BlockVisibility), bvis
        assert image_is_canonical(model)
        
        if model is None:
            return bvis
        
        newbvis = copy_visibility(bvis, zero=True)
        
        # Read wtowers-related parameters
        subgrid_size = get_parameter(kwargs, "subgrid_size")
        margin = get_parameter(kwargs, "margin")
        winc = get_parameter(kwargs, "winc")
        
        # Create an empty vis_data structure
        wtvis = wtowers.VIS_DATA()
        
        # Define the data to copy to wtvis
        antenna_count = bvis.data['uvw'].shape[1]
        bl_count = int(bvis.data.shape[0] * antenna_count * (
                    antenna_count - 1) / 2)  # bl_count = ntimes*nbases, each block will contain vis for one time count and one baseline
        # nfreq = bvis.frequency.shape[0] # the same number of frequencies for each block
        
        # Allocate memory for wtvis structure, time_count = 1, freq_count = 1
        status = wtowers.fill_vis_data_func(wtvis, antenna_count, bl_count, 1, 1)
        
        ibl = 0
        # Loop over ntime and and antennas in BlockVis, copy metadata
        for itime in range(bvis.data.shape[0]):
            for i1 in range(antenna_count):
                for i2 in range(i1 + 1, antenna_count):
                    wtvis.bl[ibl].antenna1 = i1
                    wtvis.bl[ibl].antenna2 = i2
                    wtvis.bl[ibl].time[0] = bvis.data['time'][itime]
                    wtvis.bl[ibl].freq[0] = 0.0
                    wtvis.bl[ibl].vis[0] = 0.0
                    wtvis.bl[ibl].vis[1] = 0.0
                    wtvis.bl[ibl].uvw[0] = - bvis.data['uvw'][itime, i2, i1, 0]
                    wtvis.bl[ibl].uvw[1] = bvis.data['uvw'][itime, i2, i1, 1]
                    wtvis.bl[ibl].uvw[2] = bvis.data['uvw'][itime, i2, i1, 2]
                    ibl += 1
                    # Fill stats
        status = wtowers.fill_stats_func(wtvis)
        
        # Fill wkern structure if gcfcf is provided
        wtkern = None
        if gcfcf is not None:
            wtkern = wtowers.W_KERNEL_DATA()
            wtkern = gcf2wkern(gcfcf, wtkern, model, conjugate=True)
            
            # Extracting data from BlockVisibility
        freq = bvis.frequency  # frequency, Hz
        nrows, nants, _, vnchan, vnpol = bvis.vis.shape
        
        # uvw = newbvis.data['uvw'].reshape([nrows * nants * nants, 3])
        # vis = newbvis.data['vis'].reshape([nrows * nants * nants, vnchan, vnpol])
        
        # vis[...] = 0.0 + 0.0j  # Make all vis data equal to 0 +0j
        
        # Get the image properties
        m_nchan, m_npol, ny, nx = model.data.shape
        # Check if the number of frequency channels matches in bvis and a model
        #        assert (m_nchan == v_nchan)
        assert (m_npol == vnpol)
        
        # fuvw = uvw.copy()
        # We need to flip the u and w axes. The flip in w is equivalent to the conjugation of the
        # convolution function grid_visibility to griddata
        # fuvw[:, 0] *= -1.0
        # fuvw[:, 2] *= -1.0
        
        # Find out the image size/resolution
        npixdirty = model.nwidth
        pixsize = numpy.abs(numpy.radians(model.wcs.wcs.cdelt[0]))
        
        # Define WTowers FoV in direction cosine units
        # theta = numpy.cos(numpy.pi / 2 - npixdirty * pixsize)
        theta = npixdirty * pixsize
        grid_size = nx
        
        # Make de-gridding over a frequency range and pol fields
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        for vchan in range(vnchan):
            imchan = vis_to_im[vchan]
            for vpol in range(vnpol):
                # Fill the frequency
                for ibl in range(wtvis.bl_count):
                    wtvis.bl[ibl].freq[0] = freq[vchan]
                wtvis = dirty2vis(model.data[imchan, vpol, :, :].astype(numpy.float64), grid_size, theta, wtvis,
                                  wtkern=wtkern, subgrid_size=subgrid_size, margin=margin, winc=winc)
                
                # Fill the vis and frequency data in wtvis
                ibl = 0
                # Loop over ntime and and antennas in BlockVis, copy metadata
                for itime in range(bvis.data.shape[0]):
                    for i1 in range(antenna_count):
                        for i2 in range(i1 + 1, antenna_count):
                            newbvis.data['vis'][itime, i2, i1, vchan, vpol] = wtvis.bl[ibl].vis[0] + 1j * \
                                                                              wtvis.bl[ibl].vis[1]
                            ibl += 1
        
        assert numpy.max(numpy.abs(newbvis.data['vis'])) > 0.0
        newbvis.data['vis'] = convert_pol_frame(newbvis.data['vis'], model.polarisation_frame, bvis.polarisation_frame,
                                                polaxis=4)
        assert numpy.max(numpy.abs(newbvis.data['vis'])) > 0.0

        # Now we can shift the visibility from the image frame to the original visibility frame
        return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)
    
    
    def invert_wt(bvis: BlockVisibility, model: Image, dopsf: bool = False, normalize: bool = True, gcfcf=None,
                  **kwargs) -> (Image, numpy.ndarray):
        """ Invert using py-wtowers module
        
        https://gitlab.com/ska-telescope/py-wtowers
    
        Use the image im as a template. Do PSF in a separate call.
    
        This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
        of this function. . Any shifting needed is performed here.
    
        :param bvis: BlockVisibility to be inverted
        :param im: image template (not changed)
        :param normalize: Normalize by the sum of weights (True)
        :param gcfcf:
        :param dopsf: Make a PSF
        :return: (resulting image, sum of the weights for each frequency and polarization)
    
        """
        assert image_is_canonical(model)
        
        assert isinstance(bvis, BlockVisibility), bvis
        
        im = copy_image(model)
        
        # Read wtowers-related parameters
        subgrid_size = get_parameter(kwargs, "subgrid_size")
        margin = get_parameter(kwargs, "margin")
        winc = get_parameter(kwargs, "winc")
        
        sbvis = copy_visibility(bvis)
        sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)
        
        # Create an empty vis_data structure
        wtvis = wtowers.VIS_DATA()
        
        # Define the data to copy to wtvis
        antenna_count = bvis.data['uvw'].shape[1]
        bl_count = int(bvis.data.shape[0] * antenna_count * (
                    antenna_count - 1) / 2)  # bl_count = ntimes*nbases, each block will contain vis for one time count and one baseline
        # nfreq = bvis.frequency.shape[0] # the same number of frequencies for each block
        
        # Allocate memory for wtvis structure, time_count = 1, freq_count = 1
        status = wtowers.fill_vis_data_func(wtvis, antenna_count, bl_count, 1, 1)
        
        ibl = 0
        # Loop over ntime and and antennas in BlockVis, copy metadata
        for itime in range(bvis.data.shape[0]):
            for i1 in range(antenna_count):
                for i2 in range(i1 + 1, antenna_count):
                    wtvis.bl[ibl].antenna1 = i1
                    wtvis.bl[ibl].antenna2 = i2
                    wtvis.bl[ibl].time[0] = bvis.data['time'][itime]
                    wtvis.bl[ibl].freq[0] = 0.0
                    wtvis.bl[ibl].vis[0] = 0.0
                    wtvis.bl[ibl].vis[1] = 0.0
                    wtvis.bl[ibl].uvw[0] = - bvis.data['uvw'][itime, i2, i1, 0]
                    wtvis.bl[ibl].uvw[1] = bvis.data['uvw'][itime, i2, i1, 1]
                    wtvis.bl[ibl].uvw[2] = bvis.data['uvw'][itime, i2, i1, 2]
                    ibl += 1
                    # Fill stats
        status = wtowers.fill_stats_func(wtvis)
        
        # Fill wkern structure if gcfcf is provided
        wtkern = None
        if gcfcf is not None:
            wtkern = wtowers.W_KERNEL_DATA()
            wtkern = gcf2wkern(gcfcf, wtkern, im)
        
        vis = bvis.vis
        
        freq = sbvis.frequency  # frequency, Hz
        
        nrows, nants, _, vnchan, vnpol = vis.shape
        
        # NG-related
        flags = sbvis.flags.reshape([nrows * nants * nants, vnchan, vnpol])
        uvw = sbvis.uvw.reshape([nrows * nants * nants, 3])
        ms = sbvis.flagged_vis.reshape([nrows * nants * nants, vnchan, vnpol])
        wgt = sbvis.flagged_imaging_weight.reshape([nrows * nants * nants, vnchan, vnpol])
        ##########################
        
        # Re-write for the wt vis_data structure
        if dopsf:
            ms[...] = (1 - flags).astype('complex')
        
        # NG-related
        # if epsilon > 5.0e-6:
        #    ms = ms.astype("c8")
        #    wgt = wgt.astype("f4")
        ###########################
        
        # Find out the image size/resolution
        npixdirty = im.nwidth
        pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
        
        # Define WTowers FoV in direction cosine units
        # theta = numpy.cos(numpy.pi / 2 - npixdirty * pixsize)
        theta = npixdirty * pixsize
        # Find the grid size in uvlambda
        uvlambda_init = numpy.maximum(numpy.abs(numpy.amin(bvis.data['uvw'])),
                                      numpy.abs(numpy.amax(bvis.data['uvw'])))  # m
        uvlambda_init = 2.1 * uvlambda_init
        freq_max = numpy.max(bvis.frequency)  # Hz
        clight = 299792458.  # m/s
        uvlambda_init = uvlambda_init * freq_max / clight
        uvlambda = numpy.double(npixdirty / theta)
        grid_size = int(theta * uvlambda)
        assert (uvlambda >= uvlambda_init)
        #################################
        
        nchan, npol, ny, nx = im.shape
        im.data[...] = 0.0
        sumwt = numpy.zeros([nchan, npol])
        
        ms = convert_pol_frame(ms, bvis.polarisation_frame, im.polarisation_frame, polaxis=2)
        # There's a latent problem here with the weights.
        # wgt = numpy.real(convert_pol_frame(wgt, bvis.polarisation_frame, im.polarisation_frame, polaxis=2))
        
        # Set up the conversion from visibility channels to image channels
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        for vchan in range(vnchan):
            ichan = vis_to_im[vchan]
            for pol in range(npol):
                # Nifty gridder likes to receive contiguous arrays
                # ms_1d = numpy.array([ms[row, vchan:vchan+1, pol] for row in range(nrows * nants * nants)], dtype='complex')
                # ms_1d.reshape([ms_1d.shape[0], 1])
                # wgt_1d = numpy.array([wgt[row, vchan:vchan+1, pol] for row in range(nrows * nants * nants)])
                # wgt_1d.reshape([wgt_1d.shape[0], 1])
                
                # Fill the vis and frequency data in wtvis
                ibl = 0
                # Loop over ntime and and antennas in BlockVis, copy metadata
                for itime in range(bvis.data.shape[0]):
                    for i1 in range(antenna_count):
                        for i2 in range(i1 + 1, antenna_count):
                            wtvis.bl[ibl].freq[0] = freq[vchan]
                            wtvis.bl[ibl].vis[0] = numpy.real(bvis.data['vis'][itime, i2, i1, vchan, pol])
                            wtvis.bl[ibl].vis[1] = numpy.imag(bvis.data['vis'][itime, i2, i1, vchan, pol])
                            ibl += 1
                            # Fill stats
                status = wtowers.fill_stats_func(wtvis)
                
                # Get dirty image for this frequency
                dirty = vis2dirty(grid_size, theta, wtvis, wtkern=wtkern, subgrid_size=subgrid_size, margin=margin,
                                  winc=winc)
                
                # dirty = ng.ms2dirty(
                #    fuvw, freq[vchan:vchan+1], ms_1d, wgt_1d,
                #    npixdirty, npixdirty, pixsize, pixsize, epsilon, do_wstacking=do_wstacking,
                #    nthreads=nthreads, verbosity=verbosity)
                
                sumwt[ichan, pol] += numpy.sum(wgt[:, vchan, pol])
                im.data[ichan, pol] += dirty
        
        if normalize:
            im = normalize_sumwt(im, sumwt)
        
        return im, sumwt


except ImportError:
    import warnings
    
    warnings.warn('Cannot import wtowers, wt disabled', ImportWarning)
    
    
    def predict_wt(bvis: Union[BlockVisibility, Visibility], model: Image, **kwargs) -> \
            Union[BlockVisibility, Visibility]:
        log.error("Wtowers gridder not available")
        return bvis
    
    
    def invert_wt(bvis: BlockVisibility, model: Image, dopsf: bool = False, normalize: bool = True,
                  **kwargs) -> (Image, numpy.ndarray):
        log.error("Wtowers gridder not available")
        return model, None
