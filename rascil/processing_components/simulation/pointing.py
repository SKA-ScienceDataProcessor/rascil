""" Functions for simulating pointing errors



"""

__all__ = ['simulate_gaintable_from_pointingtable', 'simulate_pointingtable_from_timeseries', 'simulate_pointingtable']

import logging

import numpy
from scipy.interpolate import RectBivariateSpline

from rascil.data_models.memory_data_models import BlockVisibility
from rascil.data_models.memory_data_models import PointingTable
from rascil.data_models.parameters import rascil_data_path
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility
from rascil.processing_components.visibility import create_visibility_from_rows, \
    calculate_blockvisibility_hourangles, calculate_blockvisibility_azel
from rascil.processing_components.visibility.iterators import vis_timeslice_iter

log = logging.getLogger('logger')


def simulate_gaintable_from_pointingtable(vis, sc, pt, vp, vis_slices=None, scale=1.0, order=3,
                                          use_radec=False, elevation_limit=15.0 * numpy.pi / 180.0,
                                          **kwargs):
    """ Create gaintables from a pointing table

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param pt: Pointing table
    :param vp: Voltage pattern in AZELGEO frame
    :param scale: Multiply the screen by this factor
    :param order: order of spline (default is 3)
    :return:
    """
    
    nant = vis.vis.shape[1]
    gaintables = [create_gaintable_from_blockvisibility(vis, **kwargs) for i in sc]
    
    nchan, npol, ny, nx = vp.data.shape
    
    real_spline = [RectBivariateSpline(range(ny), range(nx), vp.data[0, pol, ...].real, kx=order, ky=order)
                   for pol in range(npol)]
    imag_spline = [RectBivariateSpline(range(ny), range(nx), vp.data[0, pol, ...].imag, kx=order, ky=order)
                   for pol in range(npol)]
    
    if not use_radec:
        assert isinstance(vis, BlockVisibility)
        assert vp.wcs.wcs.ctype[0] == 'AZELGEO long', vp.wcs.wcs.ctype[0]
        assert vp.wcs.wcs.ctype[1] == 'AZELGEO lati', vp.wcs.wcs.ctype[1]
        
        assert vis.configuration.mount[0] == 'azel', "Mount %s not supported yet" % vis.configuration.mount[0]
        
        # The time in the Visibility is hour angle in seconds!
        number_bad = 0
        number_good = 0
        
        r2d = 180.0 / numpy.pi
        s2r = numpy.pi / 43200.0
        # For each hourangle, we need to calculate the location of a component
        # in AZELGEO. With that we can then look up the relevant gain from the
        # voltage pattern
        for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
            v = create_visibility_from_rows(vis, rows)
            ha = calculate_blockvisibility_hourangles(v).to('rad').value
            pt_rows = (pt.time == v.time)
            assert numpy.sum(pt_rows) > 0
            pointing_ha = pt.pointing[pt_rows]
            azimuth_centre, elevation_centre = calculate_blockvisibility_azel(v)
            azimuth_centre = azimuth_centre.to('rad').value
            elevation_centre = elevation_centre.to('rad').value
            
            # Calculate the az el for this hourangle and the phasecentre declination
            for icomp, comp in enumerate(sc):
                
                if elevation_centre >= elevation_limit:
                    
                    antgain = numpy.zeros([nant, npol], dtype='complex')
                    
                    # Calculate the azel of this component
                    azimuth_comp, elevation_comp = calculate_blockvisibility_azel(v, comp.direction)
                    azimuth_comp = azimuth_comp.to('rad')[0].value
                    elevation_comp = elevation_comp.to('rad')[0].value
                    
                    for ant in range(nant):
                        
                        wcs_azel = vp.wcs.deepcopy()
                        
                        az_comp = (azimuth_centre + pointing_ha[0, ant, 0, 0, 0] / numpy.cos(elevation_centre)) * r2d
                        el_comp = (elevation_centre + pointing_ha[0, ant, 0, 0, 1]) * r2d
                        
                        # We use WCS sensible coordinate handling by labelling the axes misleadingly
                        wcs_azel.wcs.crval[0] = az_comp
                        wcs_azel.wcs.crval[1] = el_comp
                        wcs_azel.wcs.ctype[0] = 'RA---SIN'
                        wcs_azel.wcs.ctype[1] = 'DEC--SIN'
                        
                        try:
                            worldloc = [azimuth_comp * r2d, elevation_comp * r2d,
                                        vp.wcs.wcs.crval[2], vp.wcs.wcs.crval[3]]
                            pixloc = wcs_azel.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                            assert pixloc[0] > 2
                            assert pixloc[0] < nx - 3
                            assert pixloc[1] > 2
                            assert pixloc[1] < ny - 3
                            for pol in range(npol):
                                gain = real_spline[pol].ev(pixloc[1], pixloc[0]) + 1j * imag_spline[pol].ev(pixloc[1],
                                                                                                            pixloc[0])
                                antgain[ant, pol] = scale * gain
                                number_good += 1
                        except (ValueError, AssertionError):
                            number_bad += 1
                            antgain[ant, :] = 1.0
                        
                        gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, :].reshape([nant, nchan, 2, 2])
                        gaintables[icomp].phasecentre = comp.direction
                else:
                    gaintables[icomp].gain[...] = 1.0 + 0.0j
                    gaintables[icomp].phasecentre = comp.direction
                    number_bad += nant
    
    else:
        assert isinstance(vis, BlockVisibility)
        assert vp.wcs.wcs.ctype[0] == 'RA---SIN', vp.wcs.wcs.ctype[0]
        assert vp.wcs.wcs.ctype[1] == 'DEC--SIN', vp.wcs.wcs.ctype[1]
        
        # The time in the Visibility is hour angle in seconds!
        number_bad = 0
        number_good = 0
        
        d2r = numpy.pi / 180.0
        ra_centre = vp.wcs.wcs.crval[0] * d2r
        dec_centre = vp.wcs.wcs.crval[1] * d2r
        
        r2d = 180.0 / numpy.pi
        s2r = numpy.pi / 43200.0
        # For each hourangle, we need to calculate the location of a component
        # in AZELGEO. With that we can then look up the relevant gain from the
        # voltage pattern
        for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
            v = create_visibility_from_rows(vis, rows)
            ha = numpy.average(v.time)
            pt_rows = (pt.time == ha)
            pointing_ha = pt.pointing[pt_rows]
            
            for icomp, comp in enumerate(sc):
                antgain = numpy.zeros([nant, npol], dtype='complex')
                antwt = numpy.zeros([nant, pol])
                # Calculate the location of the component in AZELGEO, then add the pointing offset
                # for each antenna
                ra_comp = comp.direction.ra.rad
                dec_comp = comp.direction.dec.rad
                for ant in range(nant):
                    wcs_azel = vp.wcs.deepcopy()
                    ra_pointing = (ra_centre + pointing_ha[0, ant, 0, 0, 0] / numpy.cos(dec_centre)) * r2d
                    dec_pointing = (dec_centre + pointing_ha[0, ant, 0, 0, 1]) * r2d
                    
                    # We use WCS sensible coordinate handling by labelling the axes misleadingly
                    wcs_azel.wcs.crval[0] = ra_pointing
                    wcs_azel.wcs.crval[1] = dec_pointing
                    wcs_azel.wcs.ctype[0] = 'RA---SIN'
                    wcs_azel.wcs.ctype[1] = 'DEC--SIN'
                    
                    for pol in range(npol):
                        worldloc = [ra_comp * r2d, dec_comp * r2d,
                                    vp.wcs.wcs.crval[2], vp.wcs.wcs.crval[3]]
                        try:
                            pixloc = wcs_azel.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                            assert pixloc[0] > 2
                            assert pixloc[0] < nx - 3
                            assert pixloc[1] > 2
                            assert pixloc[1] < ny - 3
                            gain = real_spline[pol].ev(pixloc[1], pixloc[0]) + 1j * imag_spline[pol].ev(pixloc[1],
                                                                                                        pixloc[0])
                            if numpy.abs(gain) > 0.0:
                                antgain[ant, pol] = 1.0 / (scale * gain)
                                antwt[ant, pol] = 1.0
                            else:
                                antgain[ant, pol] = 0.0
                                antwt[ant, pol] = 0.0
                            antwt[ant, pol] = 1.0
                            number_good += 1
                        except (ValueError, AssertionError):
                            number_bad += 1
                            antgain[ant, pol] = 1e15
                            antwt[ant, pol] = 0.0
                
                gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, :].reshape([nant, nchan, 2, 2])
                gaintables[icomp].weight[iha, :, :, :] = antwt[:, numpy.newaxis, :].reshape([nant, nchan, 2, 2])
                gaintables[icomp].phasecentre = comp.direction
    
    assert number_good > 0, "simulate_gaintable_from_pointingtable: No points inside the voltage pattern image"
    if number_bad > 0:
        log.warning(
            "simulate_gaintable_from_pointingtable: %d points are inside the voltage pattern image" % (number_good))
        log.warning(
            "simulate_gaintable_from_pointingtable: %d points are outside the voltage pattern image" % (number_bad))
    
    return gaintables


def simulate_pointingtable(pt: PointingTable, pointing_error, static_pointing_error=None, global_pointing_error=None,
                           seed=None, **kwargs) -> PointingTable:
    """ Simulate a gain table

    :type pt: PointingTable
    :param pointing_error: std of normal distribution (radians)
    :param static_pointing_error: std of normal distribution (radians)
    :param global_pointing_error: 2-vector of global pointing error (rad)
    :param kwargs:
    :return: PointingTable

    """
    
    if seed is not None:
        numpy.random.seed(seed)
    
    if static_pointing_error is None:
        static_pointing_error = [0.0, 0.0]
    
    r2s = 180.0 * 3600.0 / numpy.pi
    pt.data['pointing'] = numpy.zeros(pt.data['pointing'].shape)
    
    ntimes, nant, nchan, nrec, _ = pt.data['pointing'].shape
    if pointing_error > 0.0:
        log.debug("simulate_pointingtable: Simulating dynamic pointing error = %g (rad) %g (arcsec)"
                  % (pointing_error, r2s * pointing_error))
        
        pt.data['pointing'] += numpy.random.normal(0.0, pointing_error, pt.data['pointing'].shape)
    if (abs(static_pointing_error[0]) > 0.0) or (abs(static_pointing_error[1]) > 0.0):
        numpy.random.seed(18051955)
        log.debug("simulate_pointingtable: Simulating static pointing error = (%g, %g) (rad) (%g, %g)(arcsec)"
                  % (static_pointing_error[0], static_pointing_error[1],
                     r2s * static_pointing_error[0], r2s * static_pointing_error[1]))
        
        static_pe = numpy.zeros(pt.data['pointing'].shape[1:])
        static_pe[..., 0] = numpy.random.normal(0.0, static_pointing_error[0],
                                                static_pe[..., 0].shape)[numpy.newaxis, ...]
        static_pe[..., 1] = numpy.random.normal(0.0, static_pointing_error[1],
                                                static_pe[..., 1].shape)[numpy.newaxis, ...]
        pt.data['pointing'] += static_pe
    
    if global_pointing_error is not None:
        if seed is not None:
            numpy.random.seed(seed)
        
        log.debug("simulate_pointingtable: Simulating global pointing error = [%g, %g] (rad) [%g,s %g] (arcsec)"
                  % (global_pointing_error[0], global_pointing_error[1],
                     r2s * global_pointing_error[0], r2s * global_pointing_error[1]))
        pt.data['pointing'][..., :] += global_pointing_error
    
    return pt


def simulate_pointingtable_from_timeseries(pt, type='wind', time_series_type='precision',
                                           pointing_directory=None, reference_pointing=False,
                                           seed=None):
    """Create a pointing table with time series created from PSD.

    :param pt: Pointing table to be filled
    :param type: Type of pointing: 'tracking' or 'wind'
    :param time_series_type: Type of wind condition precision|standard|degraded
    :param pointing_directory: Name of pointing file directory
    :param reference_pointing: Use reference pointing?
    :return:
    """
    if seed is not None:
        numpy.random.seed(seed)
    
    if pointing_directory is None:
        pointing_directory = rascil_data_path("models/%s" % time_series_type)
    
    pt.data['pointing'] = numpy.zeros(pt.data['pointing'].shape)
    
    ntimes, nant, nchan, nrec, _ = pt.data['pointing'].shape
    
    # Use az and el at the beginning of this pointingtable
    axis_values = pt.nominal[0, 0, 0, 0, 0]
    el = pt.nominal[0, 0, 0, 0, 1]
    
    el_deg = el * 180.0 / numpy.pi
    az_deg = axis_values * 180.0 / numpy.pi
    
    if el_deg < 30.0:
        el_deg = 15.0
    elif el_deg < (90.0 + 45.0) / 2.0:
        el_deg = 45.0
    else:
        el_deg = 90.0
    
    if abs(az_deg) < 45.0 / 2.0:
        az_deg = 0.0
    elif abs(az_deg) < (45.0 + 90.0) / 2.0:
        az_deg = 45.0
    elif abs(az_deg) < (90.0 + 135.0) / 2.0:
        az_deg = 90.0
    elif abs(az_deg) < (135.0 + 180.0) / 2.0:
        az_deg = 135.0
    else:
        az_deg = 180.0
    
    pointing_file = '%s/El%dAz%d.dat' % (pointing_directory, int(el_deg), int(az_deg))
    log.debug("simulate_pointingtable_from_timeseries: Reading wind PSD from %s" % pointing_file)
    psd = numpy.loadtxt(pointing_file)
    
    # define some arrays
    freq = psd[:, 0]
    axesdict = {
        "az": psd[:, 1],
        "el": psd[:, 2],
        "pxel": psd[:, 3],
        "pel": psd[:, 4]
    }
    
    if type == 'tracking':
        axes = ["az", "el"]
    elif type == 'wind':
        axes = ["pxel", "pel"]
    else:
        raise ValueError("Pointing type %s not known" % type)
    
    freq_interval = 0.0001
    
    for axis in axes:
        
        axis_values = axesdict[axis]
        
        if (axis == "az") or (axis == "el"):
            # determine index of maximum PSD value; add 50 for better fit
            axis_values_max_index = numpy.argwhere(axis_values == numpy.max(axis_values))[0][0] + 50
            axis_values_max_index = min(axis_values_max_index, len(axis_values))
            # max_freq = 2.0 / pt.interval[0]
            max_freq = 0.4
            freq_max_index = numpy.argwhere(freq > max_freq)[0][0]
        else:
            break_freq = 0.01  # not max; just a break
            axis_values_max_index = numpy.argwhere(freq > break_freq)[0][0]
            # max_freq = 2.0 / pt.interval[0]
            max_freq = 0.1
            freq_max_index = numpy.argwhere(freq > max_freq)[0][0]
        
        # construct regularly-spaced frequencies
        regular_freq = numpy.arange(freq[0], freq[freq_max_index], freq_interval)
        
        regular_axis_values_max_index = numpy.argwhere(
            numpy.abs(regular_freq - freq[axis_values_max_index]) == numpy.min(
                numpy.abs(regular_freq - freq[axis_values_max_index])))[0][0]
        
        # print ('Frequency break: ', freq[az_max_index])
        # print ('Max frequency: ', max_freq)
        #
        # print ('New frequency break: ', regular_freq[regular_az_max_index])
        # print ('New max frequency: ', regular_freq[-1])
        
        if axis_values_max_index >= freq_max_index:
            raise ValueError('Frequency break is higher than highest frequency; select a lower break')
        
        # use original frequency break and max frequency to fit function
        # fit polynomial to psd up to max value
        import warnings
        from numpy import RankWarning
        warnings.simplefilter('ignore', RankWarning)
        
        p_axis_values1 = numpy.polyfit(freq[:axis_values_max_index],
                                       numpy.log(axis_values[:axis_values_max_index]), 5)
        f_axis_values1 = numpy.poly1d(p_axis_values1)
        # fit polynomial to psd beyond max value
        p_axis_values2 = numpy.polyfit(freq[axis_values_max_index:freq_max_index],
                                       numpy.log(axis_values[axis_values_max_index:freq_max_index]), 5)
        f_axis_values2 = numpy.poly1d(p_axis_values2)
        
        # use new frequency break and max frequency to apply function (ensures equal spacing of frequency intervals)
        
        # resampled to construct regularly-spaced frequencies
        regular_axis_values1 = numpy.exp(f_axis_values1(regular_freq[:regular_axis_values_max_index]))
        regular_axis_values2 = numpy.exp(f_axis_values2(regular_freq[regular_axis_values_max_index:]))
        
        # join
        regular_axis_values = numpy.append(regular_axis_values1, regular_axis_values2)
        
        m0 = len(regular_axis_values)
        
        #  check rms of resampled PSD
        # df = regular_freq[1:]-regular_freq[:-1]
        # psd2rms_pxel = numpy.sqrt(numpy.sum(regular_az[:-1]*df))
        # print ('Calculate rms of resampled PSD: ', psd2rms_pxel)
        
        original_regular_freq = regular_freq
        original_regular_axis_values = regular_axis_values
        # get amplitudes from psd values
        
        if (regular_axis_values < 0).any():
            raise ValueError('Resampling returns negative power values; change fit range')
        
        amp_axis_values = numpy.sqrt(regular_axis_values * 2 * freq_interval)
        # need to scale PSD by 2* frequency interval before square rooting, then by number of modes in resampled PSD
        
        # Now we generate some random phases
        for ant in range(nant):
            regular_freq = original_regular_freq
            regular_axis_values = original_regular_axis_values
            phi_axis_values = numpy.random.rand(len(regular_axis_values)) * 2 * numpy.pi
            # create complex array
            z_axis_values = amp_axis_values * numpy.exp(1j * phi_axis_values)  # polar
            # make symmetrical frequencies
            mirror_z_axis_values = numpy.copy(z_axis_values)
            # make complex conjugates
            mirror_z_axis_values.imag -= 2 * z_axis_values.imag
            # make negative frequencies
            mirror_regular_freq = -regular_freq
            # join
            z_axis_values = numpy.append(z_axis_values, mirror_z_axis_values[::-1])
            regular_freq = numpy.append(regular_freq, mirror_regular_freq[::-1])
            
            # add a 0 Fourier term
            zav = z_axis_values
            z_axis_values = numpy.zeros([len(zav) + 1]).astype('complex')
            z_axis_values[1:] = zav
            
            # perform inverse fft
            ts = numpy.fft.ifft(z_axis_values)
            
            # set up and check scalings
            Dt = pt.interval[0]
            ts = numpy.real(ts)
            ts *= m0  # the result is scaled by number of points in the signal, so multiply - real part - by this
            
            # The output of the iFFT will be a random time series on the finite
            # (bounded, limited) time interval t = 0 to tmax = (N-1) X Dt, #
            # where Dt = 1 / (2 X Fmax)
            
            # scale to time interval
            
            # Convert from arcsec to radians
            ts *= numpy.pi / (180.0 * 3600.0)
            
            # We take reference pointing to mean that the pointing errors are zero at the beginning
            # of the set of integrations
            if reference_pointing:
                ts[:] -= ts[0]
            
            #            pt.data['time'] = times[:ntimes]
            if axis == 'az':
                pt.data['pointing'][:, ant, :, :, 0] = ts[:ntimes, numpy.newaxis, numpy.newaxis, ...]
            elif axis == 'el':
                pt.data['pointing'][:, ant, :, :, 1] = ts[:ntimes, numpy.newaxis, numpy.newaxis, ...]
            elif axis == 'pxel':
                pt.data['pointing'][:, ant, :, :, 0] = ts[:ntimes, numpy.newaxis, numpy.newaxis, ...]
            elif axis == 'pel':
                pt.data['pointing'][:, ant, :, :, 1] = ts[:ntimes, numpy.newaxis, numpy.newaxis, ...]
            else:
                raise ValueError("Unknown axis %s" % axis)
    
    return pt
