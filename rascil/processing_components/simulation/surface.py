""" Functions for dish surface modelling

"""

__all__ = ['simulate_gaintable_from_zernikes', 'simulate_gaintable_from_voltage_pattern']

import logging

from astropy.time import Time

import numpy
from scipy.interpolate import RectBivariateSpline

from rascil.data_models.memory_data_models import BlockVisibility
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility
from rascil.processing_components.util.coordinate_support import hadec_to_azel
from rascil.processing_components.visibility import create_visibility_from_rows
from rascil.processing_components.visibility.visibility_geometry import calculate_blockvisibility_hourangles
from rascil.processing_components.util.geometry import calculate_azel
from rascil.processing_components.visibility.iterators import vis_timeslice_iter

log = logging.getLogger('logger')


def simulate_gaintable_from_voltage_pattern(vis, sc, vp, vis_slices=None, scale=1.0, order=3,
                                            use_radec=False, elevation_limit=15.0 * numpy.pi / 180.0,
                                            **kwargs):
    """ Create gaintables from a list of components and voltagr patterns

    :param elevation_limit:
    :param use_radec:
    :param vis_slices:
    :param vis:
    :param sc: Sky components for which pierce points are needed
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
        
        number_bad = 0
        number_good = 0
        
        # For each hourangle, we need to calculate the location of a component
        # in AZELGEO. With that we can then look up the relevant gain from the
        # voltage pattern
        for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
            v = create_visibility_from_rows(vis, rows)
            utc_time = Time(numpy.average(v.time)/86400.0, format='mjd', scale='utc')
            azimuth_centre, elevation_centre = calculate_azel(v.configuration.location, utc_time,
                                                              vis.phasecentre)
            azimuth_centre = azimuth_centre[0].to('deg').value
            elevation_centre = elevation_centre[0].to('deg').value
            
            # Calculate the az el for this time
            wcs_azel = vp.wcs.sub(2).deepcopy()
            
            for icomp, comp in enumerate(sc):
                
                if elevation_centre >= elevation_limit:
                    
                    antgain = numpy.zeros([nant, npol], dtype='complex')
                    antwt = numpy.zeros([nant, npol])
                    
                    # Calculate the azel of this component
                    azimuth_comp, elevation_comp = calculate_azel(v.configuration.location, utc_time,
                                                                  comp.direction)
                    cosel = numpy.cos(elevation_comp[0]).value
                    azimuth_comp = azimuth_comp[0].to('deg').value
                    elevation_comp = elevation_comp[0].to('deg').value
                    if azimuth_comp - azimuth_centre > 180.0:
                        azimuth_centre += 360.0
                    elif azimuth_comp - azimuth_centre < -180.0:
                        azimuth_centre -= 360.0

                    try:
                        worldloc = [[(azimuth_comp-azimuth_centre)*cosel, elevation_comp-elevation_centre]]
                        # radius = numpy.sqrt(((azimuth_comp-azimuth_centre)*cosel)**2 +
                        #                     (elevation_comp-elevation_centre)**2)
                        pixloc = wcs_azel.wcs_world2pix(worldloc, 1)[0]
                        assert pixloc[0] > 2
                        assert pixloc[0] < nx - 3
                        assert pixloc[1] > 2
                        assert pixloc[1] < ny - 3
                        for pol in range(npol):
                            gain = real_spline[pol].ev(pixloc[1], pixloc[0]) \
                                   + 1j * imag_spline[pol].ev(pixloc[1], pixloc[0])
                            antgain[:, pol] = gain
                        for ant in range(nant):
                            ag = antgain[ant, :].reshape([2, 2])
                            ag = numpy.linalg.inv(ag)
                            antgain[ant, :] = ag.reshape([4])
                            number_good += 1
                    except (ValueError, AssertionError):
                        number_bad += 1
                        antgain[...] = 0.0
                        antwt[...] = 0.0
                    
                    gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, :].reshape([nant, nchan, 2, 2])
                    gaintables[icomp].weight[iha, :, :, :] = antwt[:, numpy.newaxis, :].reshape([nant, nchan, 2, 2])
                    gaintables[icomp].phasecentre = comp.direction
                else:
                    gaintables[icomp].gain[...] = 1.0 + 0.0j
                    gaintables[icomp].weight[iha, :, :, :] = 0.0
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
            
            for icomp, comp in enumerate(sc):
                antgain = numpy.zeros([nant, npol], dtype='complex')
                antwt = numpy.zeros([nant, pol])
                # Calculate the location of the component in AZELGEO, then add the pointing offset
                # for each antenna
                ra_comp = comp.direction.ra.rad
                dec_comp = comp.direction.dec.rad
                for ant in range(nant):
                    wcs_azel = vp.wcs.deepcopy()
                    ra_pointing = ra_centre * r2d
                    dec_pointing = dec_centre * r2d
                    
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
    
    assert number_good > 0, "simulate_gaintable_from_voltage_pattern: No points inside the voltage pattern image"
    if number_bad > 0:
        log.warning(
            "simulate_gaintable_from_voltage_pattern: %d points are inside the voltage pattern image" % (number_good))
        log.warning(
            "simulate_gaintable_from_voltage_pattern: %d points are outside the voltage pattern image" % (number_bad))
    
    return gaintables


def simulate_gaintable_from_zernikes(vis, sc, vp_list, vp_coeffs, vis_slices=None, order=3, use_radec=False,
                                     elevation_limit=15.0 * numpy.pi / 180.0, **kwargs):
    """ Create gaintables for a set of zernikes

    :param vis:
    :param sc: Sky components for which pierce points are needed
    :param vp: List of Voltage patterns in AZELGEO frame
    :param vp_coeffs: Fractional contribution [nants, nvp]
    :param order: order of spline (default is 3)
    :return:
    """
    
    ntimes, nant = vis.vis.shape[0:2]
    vp_coeffs = numpy.array(vp_coeffs)
    gaintables = [create_gaintable_from_blockvisibility(vis, **kwargs) for i in sc]
    
    if not use_radec:
        assert isinstance(vis, BlockVisibility)
        assert vis.configuration.mount[0] == 'azel', "Mount %s not supported yet" % vis.configuration.mount[0]
        
        # The time in the Visibility is hour angle in seconds!
        number_bad = 0
        number_good = 0
        
        # Cache the splines, one per voltage pattern
        real_splines = list()
        imag_splines = list()
        for ivp, vp in enumerate(vp_list):
            assert vp.wcs.wcs.ctype[0] == 'AZELGEO long', vp.wcs.wcs.ctype[0]
            assert vp.wcs.wcs.ctype[1] == 'AZELGEO lati', vp.wcs.wcs.ctype[1]
            
            nchan, npol, ny, nx = vp.data.shape
            real_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].real, kx=order,
                                                    ky=order))
            imag_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].imag, kx=order,
                                                    ky=order))
        
        latitude = vis.configuration.location.lat.rad
        
        r2d = 180.0 / numpy.pi
        s2r = numpy.pi / 43200.0
        # For each hourangle, we need to calculate the location of a component
        # in AZELGEO. With that we can then look up the relevant gain from the
        # voltage pattern
        for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
            v = create_visibility_from_rows(vis, rows)
            ha = numpy.average(calculate_blockvisibility_hourangles(v).to('rad').value)
            
            # Calculate the az el for this hourangle and the phasecentre declination
            utc_time = Time(numpy.average(v.time)/86400.0, format='mjd', scale='utc')
            azimuth_centre, elevation_centre = calculate_azel(v.configuration.location, utc_time,
                                                              vis.phasecentre)
            azimuth_centre = azimuth_centre.to('deg').value
            elevation_centre = elevation_centre.to('deg').value
            
            for icomp, comp in enumerate(sc):
                
                if elevation_centre >= elevation_limit:
                    
                    antgain = numpy.zeros([nant], dtype='complex')
                    # Calculate the location of the component in AZELGEO, then add the pointing offset
                    # for each antenna
                    hacomp = comp.direction.ra.rad - vis.phasecentre.ra.rad + ha
                    deccomp = comp.direction.dec.rad
                    azimuth_comp, elevation_comp = hadec_to_azel(hacomp, deccomp, latitude)
                    
                    for ant in range(nant):
                        for ivp, vp in enumerate(vp_list):
                            nchan, npol, ny, nx = vp.data.shape
                            wcs_azel = vp.wcs.deepcopy()
                            
                            # We use WCS sensible coordinate handling by labelling the axes misleadingly
                            wcs_azel.wcs.crval[0] = azimuth_centre
                            wcs_azel.wcs.crval[1] = elevation_centre
                            wcs_azel.wcs.ctype[0] = 'RA---SIN'
                            wcs_azel.wcs.ctype[1] = 'DEC--SIN'
                            
                            worldloc = [azimuth_comp * r2d, elevation_comp * r2d,
                                        vp.wcs.wcs.crval[2], vp.wcs.wcs.crval[3]]
                            try:
                                pixloc = wcs_azel.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                                assert pixloc[0] > 2
                                assert pixloc[0] < nx - 3
                                assert pixloc[1] > 2
                                assert pixloc[1] < ny - 3
                                gain = real_splines[ivp].ev(pixloc[1], pixloc[0]) \
                                       + 1j * imag_splines[ivp](pixloc[1], pixloc[0])
                                antgain[ant] += vp_coeffs[ant, ivp] * gain
                                number_good += 1
                            except (ValueError, AssertionError):
                                number_bad += 1
                                antgain[ant] = 1.0
                        
                        antgain[ant] = 1.0 / antgain[ant]
                    
                    gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
                    gaintables[icomp].phasecentre = comp.direction
            else:
                gaintables[icomp].gain[...] = 1.0 + 0.0j
                gaintables[icomp].phasecentre = comp.direction
                number_bad += nant
    
    else:
        assert isinstance(vis, BlockVisibility)
        number_bad = 0
        number_good = 0
        
        # Cache the splines, one per voltage pattern
        real_splines = list()
        imag_splines = list()
        for ivp, vp in enumerate(vp_list):
            nchan, npol, ny, nx = vp.data.shape
            real_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].real, kx=order,
                                                    ky=order))
            imag_splines.append(RectBivariateSpline(range(ny), range(nx), vp.data[0, 0, ...].imag, kx=order,
                                                    ky=order))
        
        for iha, rows in enumerate(vis_timeslice_iter(vis, vis_slices=vis_slices)):
            
            # The time in the Visibility is hour angle in seconds!
            r2d = 180.0 / numpy.pi
            # For each hourangle, we need to calculate the location of a component
            # in AZELGEO. With that we can then look up the relevant gain from the
            # voltage pattern
            v = create_visibility_from_rows(vis, rows)
            ha = numpy.average(calculate_blockvisibility_hourangles(v))
            
            for icomp, comp in enumerate(sc):
                antgain = numpy.zeros([nant], dtype='complex')
                antwt = numpy.zeros([nant])
                ra_comp = comp.direction.ra.rad
                dec_comp = comp.direction.dec.rad
                for ant in range(nant):
                    for ivp, vp in enumerate(vp_list):
                        
                        assert vp.wcs.wcs.ctype[0] == 'RA---SIN', vp.wcs.wcs.ctype[0]
                        assert vp.wcs.wcs.ctype[1] == 'DEC--SIN', vp.wcs.wcs.ctype[1]
                        
                        worldloc = [ra_comp * r2d, dec_comp * r2d,
                                    vp.wcs.wcs.crval[2], vp.wcs.wcs.crval[3]]
                        nchan, npol, ny, nx = vp.data.shape
                        
                        try:
                            pixloc = vp.wcs.sub(2).wcs_world2pix([worldloc[:2]], 1)[0]
                            assert pixloc[0] > 2
                            assert pixloc[0] < nx - 3
                            assert pixloc[1] > 2
                            assert pixloc[1] < ny - 3
                            gain = real_splines[ivp].ev(pixloc[1], pixloc[0]) \
                                   + 1j * imag_splines[ivp](pixloc[1], pixloc[0])
                            antgain[ant] += vp_coeffs[ant, ivp] * gain
                            antwt[ant] = 1.0
                            number_good += 1
                        except (ValueError, AssertionError):
                            number_bad += 1
                            antgain[ant] = 1e15
                            antwt[ant] = 0.0
                        
                        antgain[ant] = 1.0 / antgain[ant]
                    
                    gaintables[icomp].gain[iha, :, :, :] = antgain[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
                    gaintables[icomp].weight[iha, :, :, :] = antwt[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
                    gaintables[icomp].phasecentre = comp.direction
    
    if number_bad > 0:
        log.warning(
            "simulate_gaintable_from_zernikes: %d points are inside the voltage pattern image" % (number_good))
        log.warning(
            "simulate_gaintable_from_zernikes: %d points are outside the voltage pattern image" % (number_bad))
    
    return gaintables
