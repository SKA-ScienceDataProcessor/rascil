"""
Base simple visibility operations, placed here to avoid circular dependencies
"""

__all__ = ['vis_summary', 'copy_visibility', 'create_visibility',
           'create_visibility_from_rows',
           'create_blockvisibility_from_ms', 'create_blockvisibility_from_uvfits',
           'create_blockvisibility', 'phaserotate_visibility',
           'export_blockvisibility_to_ms',
           'create_visibility_from_ms', 'create_visibility_from_uvfits',
           'list_ms']

import copy
import logging
import re
from typing import Union

import numpy
from astropy import units as u, constants as constants
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from rascil.data_models.memory_data_models import Visibility, BlockVisibility, \
    Configuration
from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame, \
    correlate_polarisation
from rascil.processing_components.util import skycoord_to_lmn, simulate_point
from rascil.processing_components.util import xyz_to_uvw, uvw_to_xyz, \
    hadec_to_azel
from rascil.processing_components.util.geometry import calculate_transit_time
from rascil.processing_components.visibility.visibility_geometry import calculate_blockvisibility_transit_time, \
    calculate_blockvisibility_hourangles, calculate_blockvisibility_azel

log = logging.getLogger('logger')


def vis_summary(vis: Union[Visibility, BlockVisibility]):
    """Return string summarizing the Visibility

    :param vis: Visibility or BlockVisibility
    :return: string
    """
    return "%d rows, %.3f GB" % (vis.nvis, vis.size())


def copy_visibility(vis: Union[Visibility, BlockVisibility], zero=False) -> Union[
    Visibility, BlockVisibility]:
    """Copy a visibility

    Performs a deepcopy of the data array
    :param vis: Visibility or BlockVisibility
    :returns: Visibility or BlockVisibility

    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    newvis = copy.copy(vis)
    newvis.data = numpy.copy(vis.data)
    if isinstance(vis, Visibility):
        newvis.cindex = vis.cindex
        newvis.blockvis = vis.blockvis
    if zero:
        newvis.data['vis'][...] = 0.0
    return newvis


def create_visibility(config: Configuration, times: numpy.array, frequency: numpy.array,
                      channel_bandwidth, phasecentre: SkyCoord,
                      weight: float, polarisation_frame=PolarisationFrame('stokesI'),
                      integration_time=1.0,
                      zerow=False, elevation_limit=15.0 * numpy.pi / 180.0,
                      source='unknown', meta=None,
                      utc_time=None) -> Visibility:
    """ Create a Visibility from Configuration, hour angles, and direction of source

    Note that we keep track of the integration time for BDA purposes
    
    The input times are hour angles in radians, these are converted to UTC MJD in seconds, using utc_time as
    the approximate time.



    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param frequency: frequencies (Hz] [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation (SkyCoord)
    :param channel_bandwidth: channel bandwidths: (Hz] [nchan]
    :param integration_time: Integration time ('auto' or value in s)
    :param polarisation_frame: PolarisationFrame('stokesI')
    :param integration_time: in seconds
    :param zerow: bool - set w to zero
    :param elevation_limit: in degrees
    :param source: Source name
    :param meta: Meta data as a dictionary
    :param utc_time: Time of ha definition. Default is Time("2020-01-01T00:00:00", format='isot', scale='utc')
    :return: Visibility
    """
    assert phasecentre is not None, "Must specify phase centre"
    
    if utc_time is None:
        utc_time = Time("2020-01-01T00:00:00", format='isot', scale='utc')
    
    if polarisation_frame is None:
        polarisation_frame = correlate_polarisation(config.receptor_frame)
    
    latitude = config.location.geodetic[1].to('rad').value
    
    nch = len(frequency)
    ants_xyz = config.data['xyz']
    nants = len(config.data['names'])
    nbaselines = int(nants * (nants - 1) / 2)
    ntimes = 0
    for iha, ha in enumerate(times):
        
        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            ntimes += 1
    
    npol = polarisation_frame.npol
    nrows = nbaselines * ntimes * nch
    nrowsperintegration = nbaselines * nch
    rvis = numpy.zeros([nrows, npol], dtype='complex')
    rflags = numpy.zeros([nrows, npol], dtype='int')
    rweight = numpy.ones([nrows, npol])
    rtimes = numpy.zeros([nrows])
    rfrequency = numpy.zeros([nrows])
    rchannel_bandwidth = numpy.zeros([nrows])
    rantenna1 = numpy.zeros([nrows], dtype='int')
    rantenna2 = numpy.zeros([nrows], dtype='int')
    ruvw = numpy.zeros([nrows, 3])
    
    n_flagged = 0
    
    # Do each hour angle in turn
    row = 0
    stime = calculate_transit_time(config.location, utc_time, phasecentre)
    if stime.masked:
        stime = utc_time
    for iha, ha in enumerate(times):
        
        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            rtimes[row:row + nrowsperintegration] = stime.mjd * 86400.0 + ha * 86164.1 / (2.0 * numpy.pi)
            
            # TODO: optimise loop
            # Loop over all pairs of antennas. Note that a2>a1
            ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.rad)
            for a1 in range(nants):
                for a2 in range(a1 + 1, nants):
                    rantenna1[row:row + nch] = a1
                    rantenna2[row:row + nch] = a2
                    rweight[row:row + nch, ...] = 1.0
                    rflags[row:row + nch, ...] = 0
                    
                    # Loop over all frequencies and polarisations
                    for ch in range(nch):
                        # noinspection PyUnresolvedReferences
                        k = frequency[ch] / constants.c.value
                        ruvw[row, :] = (ant_pos[a2, :] - ant_pos[a1, :]) * k
                        rfrequency[row] = frequency[ch]
                        rchannel_bandwidth[row] = channel_bandwidth[ch]
                        row += 1
    
    if zerow:
        ruvw[..., 2] = 0.0
    assert row == nrows
    rintegration_time = numpy.full_like(rtimes, integration_time)
    vis = Visibility(uvw=ruvw, time=rtimes, antenna1=rantenna1, antenna2=rantenna2,
                     frequency=rfrequency, vis=rvis, flags=rflags,
                     weight=rweight, imaging_weight=rweight,
                     integration_time=rintegration_time,
                     channel_bandwidth=rchannel_bandwidth,
                     polarisation_frame=polarisation_frame, source=source, meta=meta)
    vis.phasecentre = phasecentre
    vis.configuration = config
    log.info("create_visibility: %s" % (vis_summary(vis)))
    assert isinstance(vis, Visibility), "vis is not a Visibility: %r" % vis
    if elevation_limit is not None:
        log.info(
            'create_visibility: flagged %d/%d visibilities below elevation limit %f (rad)' %
            (n_flagged, vis.nvis, elevation_limit))
    else:
        log.debug('create_visibility: created %d visibilities' % (vis.nvis))
    
    return vis


def create_blockvisibility(config: Configuration,
                           times: numpy.array,
                           frequency: numpy.array,
                           phasecentre: SkyCoord,
                           weight: float = 1.0,
                           polarisation_frame: PolarisationFrame = None,
                           integration_time=1.0,
                           channel_bandwidth=1e6,
                           zerow=False,
                           elevation_limit=None,
                           source='unknown',
                           meta=None,
                           utc_time=None,
                           **kwargs) -> BlockVisibility:
    """ Create a BlockVisibility from Configuration, hour angles, and direction of source

    Note that we keep track of the integration time for BDA purposes
    
    The input times are hour angles in radians, these are converted to UTC MJD in seconds, using utc_time as
    the approximate time.

    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param frequency: frequencies (Hz] [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation (SkyCoord)
    :param channel_bandwidth: channel bandwidths: (Hz] [nchan]
    :param integration_time: Integration time ('auto' or value in s)
    :param polarisation_frame: PolarisationFrame('stokesI')
    :param integration_time: in seconds
    :param zerow: bool - set w to zero
    :param elevation_limit: in degrees
    :param source: Source name
    :param meta: Meta data as a dictionary
    :param utc_time: Time of ha definition default is Time("2020-01-01T00:00:00", format='isot', scale='utc')
    :return: BlockVisibility
    """
    assert phasecentre is not None, "Must specify phase centre"
    
    if utc_time is None:
        utc_time = Time("2020-01-01T00:00:00", format='isot', scale='utc')
    
    if polarisation_frame is None:
        polarisation_frame = correlate_polarisation(config.receptor_frame)
    
    latitude = config.location.geodetic[1].to('rad').value
    ants_xyz = config.data['xyz']
    nants = len(config.data['names'])
    
    ntimes = 0
    n_flagged = 0
    
    for iha, ha in enumerate(times):
        
        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            ntimes += 1
        else:
            n_flagged += 1
    
    assert ntimes > 0, "No unflagged points"
    if elevation_limit is not None:
        log.info('create_visibility: flagged %d/%d times below elevation limit %f (rad)' %
                 (n_flagged, ntimes, elevation_limit))
    else:
        log.debug('create_blockvisibility: created %d times' % (ntimes))
    
    npol = polarisation_frame.npol
    nchan = len(frequency)
    visshape = [ntimes, nants, nants, nchan, npol]
    rvis = numpy.zeros(visshape, dtype='complex')
    rflags = numpy.zeros(visshape, dtype='int')
    rweight = numpy.ones(visshape)
    rimaging_weight = numpy.ones(visshape)
    rtimes = numpy.zeros([ntimes])
    rintegrationtime = numpy.zeros([ntimes])
    ruvw = numpy.zeros([ntimes, nants, nants, 3])
    
    # Do each hour angle in turn
    itime = 0
    stime = calculate_transit_time(config.location, utc_time, phasecentre)
    if stime.masked:
        stime = utc_time
    
    for iha, ha in enumerate(times):
        
        # Calculate the positions of the antennas as seen for this hour angle
        # and declination
        ant_pos = xyz_to_uvw(ants_xyz, ha, phasecentre.dec.rad)
        _, elevation = hadec_to_azel(ha, phasecentre.dec.rad, latitude)
        if elevation_limit is None or (elevation > elevation_limit):
            rtimes[itime] = stime.mjd * 86400.0 + ha * 86164.1 / (2.0 * numpy.pi)
            rweight[itime, ...] = 1.0
            rflags[itime, ...] = 0
            
            # Loop over all pairs of antennas. Note that a2>a1
            for a1 in range(nants):
                rweight[itime, a1, a1, ...] = 0.0
                rflags[itime, a1, a1, ...] = 1.0
                for a2 in range(a1 + 1, nants):
                    ruvw[itime, a2, a1, :] = (ant_pos[a2, :] - ant_pos[a1, :])
                    ruvw[itime, a1, a2, :] = (ant_pos[a1, :] - ant_pos[a2, :])
            if itime > 0:
                rintegrationtime[itime] = rtimes[itime] - rtimes[itime - 1]
            itime += 1
    
    if itime > 1:
        rintegrationtime[0] = rintegrationtime[1]
    else:
        rintegrationtime[0] = integration_time
    rchannel_bandwidth = channel_bandwidth
    if zerow:
        ruvw[..., 2] = 0.0
    vis = BlockVisibility(uvw=ruvw, time=rtimes, frequency=frequency, vis=rvis,
                          weight=rweight,
                          imaging_weight=rimaging_weight, flags=rflags,
                          integration_time=rintegrationtime,
                          channel_bandwidth=rchannel_bandwidth,
                          polarisation_frame=polarisation_frame, source=source, meta=meta)
    vis.phasecentre = phasecentre
    vis.configuration = config
    log.debug("create_blockvisibility: %s" % (vis_summary(vis)))
    assert isinstance(vis, BlockVisibility), "vis is not a BlockVisibility: %r" % vis
    
    return vis


def create_visibility_from_rows(vis: Union[Visibility, BlockVisibility],
                                rows: numpy.ndarray, makecopy=True):
    """ Create a Visibility from selected rows

    :param vis: Visibility or BlockVisibility
    :param rows: Boolean array of row selction
    :param makecopy: Make a deep copy (True)
    :return: Visibility or BlockVisibility
    """
    
    if rows is None or numpy.sum(rows) == 0:
        return None
    
    assert len(
        rows) == vis.nvis, "Length of rows does not agree with length of visibility"
    
    if isinstance(vis, Visibility):
        
        if makecopy:
            newvis = copy_visibility(vis)
            if vis.cindex is not None and len(rows) == len(vis.cindex):
                newvis.cindex = vis.cindex[rows]
            else:
                newvis.cindex = None
            if vis.blockvis is not None:
                newvis.blockvis = vis.blockvis
            newvis.data = copy.deepcopy(vis.data[rows])
            return newvis
        else:
            vis.data = copy.deepcopy(vis.data[rows])
            if vis.cindex is not None:
                vis.cindex = vis.cindex[rows]
            return vis
    else:
        
        if makecopy:
            newvis = copy_visibility(vis)
            newvis.data = copy.deepcopy(vis.data[rows])
            return newvis
        else:
            vis.data = copy.deepcopy(vis.data[rows])
            
            return vis


def phaserotate_visibility(vis: Union[Visibility, BlockVisibility],
                           newphasecentre: SkyCoord, tangent=True,
                           inverse=False) -> Union[Visibility, BlockVisibility]:
    """ Phase rotate from the current phase centre to a new phase centre

    If tangent is False the uvw are recomputed and the visibility phasecentre is updated.
    Otherwise only the visibility phases are adjusted

    :param vis: Visibility or BlockVisibility to be rotated
    :param newphasecentre: SkyCoord of new phasecentre
    :param tangent: Stay on the same tangent plane? (True)
    :param inverse: Actually do the opposite
    :return: Visibility or BlockVisibility
    """
    l, m, n = skycoord_to_lmn(newphasecentre, vis.phasecentre)
    
    # No significant change?
    if numpy.abs(n) < 1e-15:
        return vis
    
    # Make a new copy
    newvis = copy_visibility(vis)
    
    if isinstance(vis, Visibility):
        phasor = calculate_visibility_phasor(newphasecentre, newvis)
        
        if inverse:
            newvis.data['vis'] *= phasor
        else:
            newvis.data['vis'] *= numpy.conj(phasor)
        
        # To rotate UVW, rotate into the global XYZ coordinate system and back. We have the option of
        # staying on the tangent plane or not. If we stay on the tangent then the raster will
        # join smoothly at the edges. If we change the tangent then we will have to reproject to get
        # the results on the same image, in which case overlaps or gaps are difficult to deal with.
        if not tangent:
            if inverse:
                xyz = uvw_to_xyz(vis.data['uvw'], ha=-newvis.phasecentre.ra.rad,
                                 dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = \
                    xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad,
                               dec=newphasecentre.dec.rad)[...]
            else:
                # This is the original (non-inverse) code
                xyz = uvw_to_xyz(newvis.data['uvw'], ha=-newvis.phasecentre.ra.rad,
                                 dec=newvis.phasecentre.dec.rad)
                newvis.data['uvw'][...] = \
                    xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad,
                               dec=newphasecentre.dec.rad)[...]
            newvis.phasecentre = newphasecentre
        return newvis
    
    elif isinstance(vis, BlockVisibility):
        
        phasor = calculate_blockvisibility_phasor(newphasecentre, newvis)
        
        if inverse:
            newvis.data['vis'] *= phasor
        else:
            newvis.data['vis'] *= numpy.conj(phasor)
        
        # To rotate UVW, rotate into the global XYZ coordinate system and back. We have the option of
        # staying on the tangent plane or not. If we stay on the tangent then the raster will
        # join smoothly at the edges. If we change the tangent then we will have to reproject to get
        # the results on the same image, in which case overlaps or gaps are difficult to deal with.
        if not tangent:
            # UVW is shape [nants, nants, 3], we want [nants * nants, 3]
            nrows, nants, _, _ = vis.uvw.shape
            uvw_linear = vis.uvw.reshape([nrows * nants * nants, 3])
            if inverse:
                xyz = uvw_to_xyz(uvw_linear, ha=-newvis.phasecentre.ra.rad,
                                 dec=newvis.phasecentre.dec.rad)
                uvw_linear = \
                    xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad,
                               dec=newphasecentre.dec.rad)[...]
            else:
                # This is the original (non-inverse) code
                xyz = uvw_to_xyz(uvw_linear, ha=-newvis.phasecentre.ra.rad,
                                 dec=newvis.phasecentre.dec.rad)
                uvw_linear = \
                    xyz_to_uvw(xyz, ha=-newphasecentre.ra.rad,
                               dec=newphasecentre.dec.rad)[...]
            newvis.phasecentre = newphasecentre
            newvis.data['uvw'][...] = uvw_linear.reshape([nrows, nants, nants, 3])
        return newvis
    else:
        raise ValueError("vis argument neither Visibility or BlockVisibility")


def export_blockvisibility_to_ms(msname, vis_list, source_name=None):
    """ Minimal BlockVisibility to MS converter

    The MS format is much more general than the RASCIL BlockVisibility so we cut many corners. This requires casacore to be
    installed. If not an exception ModuleNotFoundError is raised.

    Write a list of BlockVisibility's to a MS file, split by field and spectral window

    :param msname: File name of MS
    :param vis_list: list of BlockVisibility
    :param source_name: Source name to use
    :param ack: Ask casacore to acknowledge each table operation
    :return:
    """
    try:
        import casacore.tables.tableutil as pt
        from casacore.tables import (makescacoldesc, makearrcoldesc, table, maketabdesc,
                                     tableexists, tableiswritable,
                                     tableinfo, tablefromascii, tabledelete, makecoldesc,
                                     msconcat, removeDerivedMSCal,
                                     taql, tablerename, tablecopy, tablecolumn,
                                     addDerivedMSCal, removeImagingColumns,
                                     addImagingColumns, required_ms_desc,
                                     tabledefinehypercolumn, default_ms,
                                     makedminfo,
                                     default_ms_subtable)
        from rascil.processing_components.visibility.msv2fund import Antenna, Stand
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")
    
    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")
    
    # log.debug("create_blockvisibility_from_ms: %s" % str(tab.info()))
    # Start the table
    tbl = msv2.Ms(msname, ref_time=0, source_name=source_name, if_delete=True)
    for vis in vis_list:
        if source_name is None:
            source_name = vis.source
        # Check polarisation
        npol = vis.npol
        nchan = vis.nchan
        if vis.polarisation_frame.type == 'linear':
            polarization = ['XX', 'XY', 'YX', 'YY']
        elif vis.polarisation_frame.type == 'linearnp':
            polarization = ['XX', 'YY']
        elif vis.polarisation_frame.type == 'stokesI':
            polarization = ['XX']
        elif vis.polarisation_frame.type == 'circular':
            polarization = ['RR', 'RL', 'LR', 'LL']
        elif vis.polarisation_frame.type == 'circularnp':
            polarization = ['RR', 'LL']
        elif vis.polarisation_frame.type == 'stokesIQUV':
            polarization = ['I', 'Q', 'U', 'V']
        elif vis.polarisation_frame.type == 'stokesIQ':
            polarization = ['I', 'Q']
        elif vis.polarisation_frame.type == 'stokesIV':
            polarization = ['I', 'V']
        else:
            raise ValueError(
                "Unknown visibility polarisation %s" % (vis.polarisation_frame.type))
        # Current RASCIL supports I
        tbl.set_stokes(polarization)
        tbl.set_frequency(vis.frequency, vis.channel_bandwidth)
        n_ant = len(vis.configuration.xyz)
        
        antennas = []
        names = vis.configuration.names
        xyz = vis.configuration.xyz
        for i in range(len(names)):
            antennas.append(Antenna(i, Stand(names[i], xyz[i, 0], xyz[i, 1], xyz[i, 2])))
        
        # Set baselines and data
        bl_list = []
        
        antennas2 = antennas
        
        for i in range(0, n_ant - 1):
            for j in range(i + 1, n_ant):
                bl_list.append((antennas[i], antennas2[j]))
        
        tbl.set_geometry(vis.configuration, antennas)
        nbaseline = len(bl_list)
        ntimes = len(vis.data['time'])
        
        ms_vis = numpy.zeros([ntimes, nbaseline, nchan, npol]).astype('complex')
        ms_uvw = numpy.zeros([ntimes, nbaseline, 3])
        time = vis.data['time']
        int_time = vis.data['integration_time']
        bv_vis = vis.data['vis']
        bv_uvw = vis.data['uvw']
        
        for row, _ in enumerate(time):
            # MS has shape [row, npol, nchan]
            # BV has shape [ntimes, nants, nants, nchan, npol]
            bl = 0
            for i in range(0, n_ant - 1):
                for j in range(i + 1, n_ant):
                    ms_vis[row, bl, ...] = bv_vis[row, j, i, ...]
                    ms_uvw[row, bl, :] = bv_uvw[row, j, i, :]
                    bl += 1
        
        for ntime, time in enumerate(vis.data['time']):
            for ipol, pol in enumerate(polarization):
                if int_time[ntime] is not None:
                    tbl.add_data_set(time, int_time[ntime], bl_list,
                                     ms_vis[ntime, ..., ipol], pol=pol,
                                     source=source_name,
                                     phasecentre=vis.phasecentre, uvw=ms_uvw[ntime, :, :])
                else:
                    tbl.add_data_set(time, 0, bl_list, ms_vis[ntime, ..., ipol], pol=pol,
                                     source=source_name, phasecentre=vis.phasecentre,
                                     uvw=ms_uvw[ntime, :, :])
    tbl.write()


def list_ms(msname, ack=False):
    """ List sources and data descriptors in a MeasurementSet

    :param msname: File name of MS
    :param ack: Ask casacore to acknowledge each table operation
    :return: sources, data descriptors

    For example::
        print(list_ms('3C277.1_avg.ms'))
        (['1302+5748', '0319+415', '1407+284', '1252+5634', '1331+305'], [0, 1, 2, 3])
    """
    try:
        from casacore.tables import table  # pylint: disable=import-error
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")
    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")
    
    tab = table(msname, ack=ack)
    log.debug("list_ms: %s" % str(tab.info()))
    
    fieldtab = table('%s/FIELD' % msname, ack=False)
    sources = fieldtab.getcol('NAME')
    
    ddtab = table('%s/DATA_DESCRIPTION' % msname, ack=False)
    dds = list(range(ddtab.nrows()))
    
    return sources, dds


def create_blockvisibility_from_ms(msname, channum=None, start_chan=None, end_chan=None,
                                   ack=False,
                                   datacolumn='DATA', selected_sources=None,
                                   selected_dds=None):
    """ Minimal MS to BlockVisibility converter

    The MS format is much more general than the RASCIL BlockVisibility so we cut many corners.
    This requires casacore to be installed. If not an exception ModuleNotFoundError is raised.

    Creates a list of BlockVisibility's, split by field and spectral window
    
    Reading of a subset of channels is possible using either start_chan and end_chan or channnum. Using start_chan 
    and end_chan is preferred since it only reads the channels required. Channum is more flexible and can be used to
    read a random list of channels.
    
    :param msname: File name of MS
    :param channum: range of channels e.g. range(17,32), default is None meaning all
    :param start_chan: Starting channel to read
    :param end_chan: End channel to read
    :param ack: Ask casacore to acknowledge each table operation
    :param datacolumn: MS data column to read DATA, CORRECTED_DATA, or MODEL_DATA
    :param selected_sources: Sources to select
    :param selected_dds: Data descriptors to select
    :return: List of BlockVisibility

    For example::

        selected_sources = ['1302+5748', '1252+5634']
        bvis_list = create_blockvisibility_from_ms('../../data/3C277.1_avg.ms', datacolumn='CORRECTED_DATA',
                                           selected_sources=selected_sources)
        sources = numpy.unique([bv.source for bv in bvis_list])
        print(sources)
        ['1252+5634' '1302+5748']

    """
    try:
        from casacore.tables import table  # pylint: disable=import-error
    except ModuleNotFoundError:
        raise ModuleNotFoundError("casacore is not installed")
    try:
        from rascil.processing_components.visibility import msv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("cannot import msv2")
    
    tab = table(msname, ack=ack)
    log.debug("create_blockvisibility_from_ms: %s" % str(tab.info()))
    
    if selected_sources is None:
        fields = numpy.unique(tab.getcol('FIELD_ID'))
    else:
        fieldtab = table('%s/FIELD' % msname, ack=False)
        sources = fieldtab.getcol('NAME')
        fields = list()
        for field, source in enumerate(sources):
            if source in selected_sources: fields.append(field)
        assert len(fields) > 0, "No sources selected"
    
    if selected_dds is None:
        dds = numpy.unique(tab.getcol('DATA_DESC_ID'))
    else:
        dds = selected_dds
    
    log.debug(
        "create_blockvisibility_from_ms: Reading unique fields %s, unique data descriptions %s" % (
            str(fields), str(dds)))
    vis_list = list()
    for field in fields:
        ftab = table(msname, ack=ack).query('FIELD_ID==%d' % field, style='')
        assert ftab.nrows() > 0, "Empty selection for FIELD_ID=%d" % (field)
        for dd in dds:
            # Now get info from the subtables
            ddtab = table('%s/DATA_DESCRIPTION' % msname, ack=False)
            spwid = ddtab.getcol('SPECTRAL_WINDOW_ID')[dd]
            polid = ddtab.getcol('POLARIZATION_ID')[dd]
            ddtab.close()
            
            meta = {'MSV2': {'FIELD_ID': field, 'DATA_DESC_ID': dd}}
            ms = ftab.query('DATA_DESC_ID==%d' % dd, style='')
            assert ms.nrows() > 0, "Empty selection for FIELD_ID=%d and DATA_DESC_ID=%d" % (field, dd)
            log.debug("create_blockvisibility_from_ms: Found %d rows" % (ms.nrows()))
            # The TIME column has descriptor:
            # {'valueType': 'double', 'dataManagerType': 'IncrementalStMan', 'dataManagerGroup': 'TIME',
            # 'option': 0, 'maxlen': 0, 'comment': 'Modified Julian Day',
            # 'keywords': {'QuantumUnits': ['s'], 'MEASINFO': {'type': 'epoch', 'Ref': 'UTC'}}}
            otime = ms.getcol('TIME')
            datacol = ms.getcol(datacolumn, nrow=1)
            datacol_shape = list(datacol.shape)
            channels = datacol.shape[-2]
            log.debug("create_blockvisibility_from_ms: Found %d channels" % (channels))
            if channum is None:
                if start_chan is not None and end_chan is not None:
                    try:
                        log.debug(
                            "create_blockvisibility_from_ms: Reading channels from %d to %d" %
                            (start_chan, end_chan))
                        blc = [start_chan, 0]
                        trc = [end_chan, datacol_shape[-1] - 1]
                        channum = range(start_chan, end_chan + 1)
                        ms_vis = ms.getcolslice(datacolumn, blc=blc, trc=trc)
                        ms_flags = ms.getcolslice('FLAG', blc=blc, trc=trc)
                        ms_weight = ms.getcol('WEIGHT')
                    except IndexError:
                        raise IndexError("channel number exceeds max. within ms")
                
                else:
                    log.debug(
                        "create_blockvisibility_from_ms: Reading all %d channels" % (
                            channels))
                    try:
                        channum = range(channels)
                        ms_vis = ms.getcol(datacolumn)[:, channum, :]
                        ms_weight = ms.getcol('WEIGHT')
                        ms_flags = ms.getcol('FLAG')
                        channum = range(channels)
                    except IndexError:
                        raise IndexError("channel number exceeds max. within ms")
            else:
                log.debug(
                    "create_blockvisibility_from_ms: Reading channels %s " % (channum))
                channum = range(channels)
                try:
                    ms_vis = ms.getcol(datacolumn)[:, channum, :]
                    ms_flags = ms.getcol('FLAG')[:, channum, :]
                    ms_weight = ms.getcol('WEIGHT')[:, :]
                except IndexError:
                    raise IndexError("channel number exceeds max. within ms")
            
            uvw = -1 * ms.getcol('UVW')
            antenna1 = ms.getcol('ANTENNA1')
            antenna2 = ms.getcol('ANTENNA2')
            integration_time = ms.getcol('INTERVAL')
            
            time = (otime - integration_time / 2.0)
            
            start_time = numpy.min(time) / 86400.0
            end_time = numpy.max(time) / 86400.0
            
            log.debug("create_blockvisibility_from_ms: Observation from %s to %s" %
                      (Time(start_time, format='mjd').iso,
                       Time(end_time, format='mjd').iso))
            
            spwtab = table('%s/SPECTRAL_WINDOW' % msname, ack=False)
            cfrequency = spwtab.getcol('CHAN_FREQ')[spwid][channum]
            cchannel_bandwidth = spwtab.getcol('CHAN_WIDTH')[spwid][channum]
            nchan = cfrequency.shape[0]
            
            # Get polarisation info
            poltab = table('%s/POLARIZATION' % msname, ack=False)
            corr_type = poltab.getcol('CORR_TYPE')[polid]
            corr_type = sorted(corr_type)
            # These correspond to the CASA Stokes enumerations
            if numpy.array_equal(corr_type, [1, 2, 3, 4]):
                polarisation_frame = PolarisationFrame('stokesIQUV')
                npol = 4
            elif numpy.array_equal(corr_type, [1, 2]):
                polarisation_frame = PolarisationFrame('stokesIQ')
                npol = 2
            elif numpy.array_equal(corr_type, [1, 4]):
                polarisation_frame = PolarisationFrame('stokesIV')
                npol = 2
            elif numpy.array_equal(corr_type, [5, 6, 7, 8]):
                polarisation_frame = PolarisationFrame('circular')
                npol = 4
            elif numpy.array_equal(corr_type, [5, 8]):
                polarisation_frame = PolarisationFrame('circularnp')
                npol = 2
            elif numpy.array_equal(corr_type, [9, 10, 11, 12]):
                polarisation_frame = PolarisationFrame('linear')
                npol = 4
            elif numpy.array_equal(corr_type, [9, 12]):
                polarisation_frame = PolarisationFrame('linearnp')
                npol = 2
            elif numpy.array_equal(corr_type, [9]):
                npol = 1
                polarisation_frame = PolarisationFrame('stokesI')
            else:
                raise KeyError("Polarisation not understood: %s" % str(corr_type))
            
            # Get configuration
            anttab = table('%s/ANTENNA' % msname, ack=False)
            names = numpy.array(anttab.getcol('NAME'))
            
            ant_map = list()
            actual = 0
            for i, name in enumerate(names):
                if name != "":
                    ant_map.append(actual)
                    actual += 1
                else:
                    ant_map.append(-1)
            
            mount = numpy.array(anttab.getcol('MOUNT'))[names != '']
            diameter = numpy.array(anttab.getcol('DISH_DIAMETER'))[names != '']
            xyz = numpy.array(anttab.getcol('POSITION'))[names != '']
            names = numpy.array(anttab.getcol('NAME'))[names != '']
            nants = len(names)
            
            antenna1 = list(map(lambda i: ant_map[i], antenna1))
            antenna2 = list(map(lambda i: ant_map[i], antenna2))
            
            configuration = Configuration(name='', data=None, location=None,
                                          names=names, xyz=xyz, mount=mount, frame="geocentric",
                                          receptor_frame=ReceptorFrame("linear"),
                                          diameter=diameter)
            # Get phasecentres
            fieldtab = table('%s/FIELD' % msname, ack=False)
            pc = fieldtab.getcol('PHASE_DIR')[field, 0, :]
            source = fieldtab.getcol('NAME')[field]
            phasecentre = SkyCoord(ra=pc[0] * u.rad, dec=pc[1] * u.rad, frame='icrs',
                                   equinox='J2000')
            
            time_index_row = numpy.zeros_like(time, dtype='int')
            time_last = time[0]
            time_index = 0
            for row, _ in enumerate(time):
                if time[row] > time_last + integration_time[row]:
                    assert time[row] > time_last, "MS is not time-sorted - cannot convert"
                    time_index += 1
                    time_last = time[row]
                time_index_row[row] = time_index
            
            ntimes = time_index + 1
            
            bv_times = numpy.zeros([ntimes])
            bv_vis = numpy.zeros([ntimes, nants, nants, nchan, npol]).astype('complex')
            bv_flags = numpy.zeros([ntimes, nants, nants, nchan, npol]).astype('int')
            bv_weight = numpy.zeros([ntimes, nants, nants, nchan, npol])
            bv_imaging_weight = numpy.zeros([ntimes, nants, nants, nchan, npol])
            bv_uvw = numpy.zeros([ntimes, nants, nants, 3])
            bv_integration_time = numpy.zeros([ntimes])
            
            for row, _ in enumerate(time):
                time_index = time_index_row[row]
                bv_times[time_index] = time[row]
                bv_vis[time_index, antenna2[row], antenna1[row], ...] = ms_vis[row, ...]
                bv_flags[time_index, antenna2[row], antenna1[row], ...] = ms_flags[
                    row, ...]
                bv_weight[time_index, antenna2[row], antenna1[row], :, ...] = ms_weight[
                    row, numpy.newaxis, ...]
                bv_imaging_weight[time_index, antenna2[row], antenna1[row], :, ...] = \
                    ms_weight[row, numpy.newaxis, ...]
                bv_uvw[time_index, antenna2[row], antenna1[row], :] = uvw[row, :]
                bv_integration_time[time_index] = integration_time[row]
            
            vis_list.append(BlockVisibility(uvw=bv_uvw,
                                            time=bv_times,
                                            frequency=cfrequency,
                                            channel_bandwidth=cchannel_bandwidth,
                                            vis=bv_vis,
                                            flags=bv_flags,
                                            weight=bv_weight,
                                            integration_time=bv_integration_time,
                                            imaging_weight=bv_imaging_weight,
                                            configuration=configuration,
                                            phasecentre=phasecentre,
                                            polarisation_frame=polarisation_frame,
                                            source=source, meta=meta))
        tab.close()
    return vis_list


def create_visibility_from_ms(msname, channum=None, start_chan=None, end_chan=None,
                              ack=False):
    """ Minimal MS to BlockVisibility converter

    The MS format is much more general than the RASCIL BlockVisibility so we cut many corners. This requires casacore to be
    installed. If not an exception ModuleNotFoundError is raised.

    Creates a list of BlockVisibility's, split by field and spectral window

    Reading of a subset of channels is possible using either start_chan and end_chan or channnum. Using start_chan
    and end_chan is preferred since it only reads the channels required. Channum is more flexible and can be used to
    read a random list of channels.
    
    :param msname: File name of MS
    :param channum: range of channels e.g. range(17,32), default is None meaning all
    :param start_chan: Starting channel to read
    :param end_chan: End channel to read
    :return:
    """
    from rascil.processing_components.visibility.coalesce import \
        convert_blockvisibility_to_visibility
    return [convert_blockvisibility_to_visibility(v)
            for v in create_blockvisibility_from_ms(msname=msname, channum=channum,
                                                    start_chan=start_chan,
                                                    end_chan=end_chan, ack=ack)]


def create_blockvisibility_from_uvfits(fitsname, channum=None, ack=False, antnum=None):
    """ Minimal UVFIT to BlockVisibility converter

    The UVFITS format is much more general than the RASCIL BlockVisibility so we cut many corners.
    
    Creates a list of BlockVisibility's, split by field and spectral window
    
    :param fitsname: File name of UVFITS
    :param channum: range of channels e.g. range(17,32), default is None meaning all
    :param antnum: the number of antenna
    :return:
    """
    
    def find_time_slots(times):
        """ Find the time slots
        
        :param times:
        :return:
        """
        intervals = times[1:] - times[0:-1]
        integration_time = numpy.median(intervals[intervals > 0.0])
        last_time = times[0]
        time_slots = list()
        for t in times:
            if t > last_time + integration_time:
                last_time = t
                time_slots.append(last_time)
        
        time_slots = numpy.array(time_slots)
        
        return time_slots
    
    def param_dict(hdul):
        "Return the dictionary of the random parameters"
        
        """
        The keys of the dictionary are the parameter names uppercased for
        consistency. The values are the column numbers.

        If multiple parameters have the same name (e.g., DATE) their
        columns are entered as a list.
        """
        
        pre = re.compile(r"PTYPE(?P<i>\d+)")
        res = {}
        for k, v in hdul.header.items():
            m = pre.match(k)
            if m:
                vu = v.upper()
                if vu in res:
                    res[vu] = [res[vu], int(m.group("i"))]
                else:
                    res[vu] = int(m.group("i"))
        return res
    
    # Open the file
    with fits.open(fitsname) as hdul:
        
        # Read Spectral Window
        nspw = hdul[0].header['NAXIS5']
        # Read Channel and Frequency Interval
        freq_ref = hdul[0].header['CRVAL4']
        delt_freq = hdul[0].header['CDELT4']
        # Real the number of channels in one spectral window
        channels = hdul[0].header['NAXIS4']
        freq = numpy.zeros([nspw, channels])
        # Read Frequency or IF
        freqhdulname = "AIPS FQ"
        sdhu = hdul.index_of(freqhdulname)
        if_freq = hdul[sdhu].data['IF FREQ'].ravel()
        for i in range(nspw):
            temp = numpy.array(
                [if_freq[i] + freq_ref + delt_freq * ff for ff in range(channels)])
            freq[i, :] = temp[:]
        freq_delt = numpy.ones(channels) * delt_freq
        if channum is None:
            channum = range(channels)
        
        # Read time. We are trying to find a discrete set of times to use in
        # BlockVisibility.
        bvtimes = Time(hdul[0].data['DATE'], hdul[0].data['_DATE'], format='jd')
        bv_times = find_time_slots(bvtimes.jd)
        
        ntimes = len(bv_times)
        
        # # Get Antenna
        # blin = hdul[0].data['BASELINE']
        antennahdulname = "AIPS AN"
        adhu = hdul.index_of(antennahdulname)
        try:
            antenna_name = hdul[adhu].data['ANNAME']
            antenna_name = antenna_name.encode('ascii', 'ignore')
        except ValueError:
            antenna_name = None
        
        antenna_xyz = hdul[adhu].data['STABXYZ']
        antenna_mount = hdul[adhu].data['MNTSTA']
        try:
            antenna_diameter = hdul[adhu].data['DIAMETER']
        except (ValueError, KeyError):
            antenna_diameter = None
        # To reading some UVFITS with wrong numbers of antenna
        if antnum is not None and antenna_name is not None:
            antenna_name = antenna_name[:antnum]
            antenna_xyz = antenna_xyz[:antnum]
            antenna_mount = antenna_mount[:antnum]
            if antenna_diameter is not None:
                antenna_diameter = antenna_diameter[:antnum]
        
        nants = len(antenna_xyz)
        
        # Get polarisation info
        npol = hdul[0].header['NAXIS3']
        corr_type = numpy.arange(hdul[0].header['NAXIS3']) - (
                hdul[0].header['CRPIX3'] - 1)
        corr_type *= hdul[0].header['CDELT3']
        corr_type += hdul[0].header['CRVAL3']
        # xx yy xy yx
        # These correspond to the CASA Stokes enumerations
        if numpy.array_equal(corr_type, [1, 2, 3, 4]):
            polarisation_frame = PolarisationFrame('stokesIQUV')
        elif numpy.array_equal(corr_type, [1, 4]):
            polarisation_frame = PolarisationFrame('stokesIV')
        elif numpy.array_equal(corr_type, [1, 2]):
            polarisation_frame = PolarisationFrame('stokesIQ')
        elif numpy.array_equal(corr_type, [-1, -2, -3, -4]):
            polarisation_frame = PolarisationFrame('circular')
        elif numpy.array_equal(corr_type, [-1, -4]):
            polarisation_frame = PolarisationFrame('circularnp')
        elif numpy.array_equal(corr_type, [-5, -6, -7, -8]):
            polarisation_frame = PolarisationFrame('linear')
        elif numpy.array_equal(corr_type, [-5, -8]):
            polarisation_frame = PolarisationFrame('linearnp')
        else:
            raise KeyError("Polarisation not understood: %s" % str(corr_type))
        
        configuration = Configuration(name='', data=None, location=None,
                                      names=antenna_name, xyz=antenna_xyz,
                                      mount=antenna_mount, frame=None,
                                      receptor_frame=polarisation_frame,
                                      diameter=antenna_diameter)
        
        # Get RA and DEC
        phase_center_ra_degrees = numpy.float(hdul[0].header['CRVAL6'])
        phase_center_dec_degrees = numpy.float(hdul[0].header['CRVAL7'])
        
        # Get phasecentres
        phasecentre = SkyCoord(ra=phase_center_ra_degrees * u.deg,
                               dec=phase_center_dec_degrees * u.deg, frame='icrs',
                               equinox='J2000')
        
        # Get UVW
        d = param_dict(hdul[0])
        if "UU" in d:
            uu = hdul[0].data['UU']
            vv = hdul[0].data['VV']
            ww = hdul[0].data['WW']
        else:
            uu = hdul[0].data['UU---SIN']
            vv = hdul[0].data['VV---SIN']
            ww = hdul[0].data['WW---SIN']
        _vis = hdul[0].data['DATA']
        
        row = 0
        nchan = len(channum)
        vis_list = list()
        for spw_index in range(nspw):
            bv_vis = numpy.zeros([ntimes, nants, nants, nchan, npol]).astype('complex')
            bv_flags = numpy.zeros([ntimes, nants, nants, nchan, npol]).astype('int')
            bv_weight = numpy.zeros([ntimes, nants, nants, nchan, npol])
            bv_uvw = numpy.zeros([ntimes, nants, nants, 3])
            for time_index, time in enumerate(bv_times):
                for antenna1 in range(nants - 1):
                    for antenna2 in range(antenna1 + 1, nants):
                        for channel_no, channel_index in enumerate(channum):
                            for pol_index in range(npol):
                                bv_vis[
                                    time_index, antenna2, antenna1, channel_no, pol_index] = complex(
                                    _vis[row, :, :, spw_index, channel_index, pol_index,
                                    0],
                                    _vis[row, :, :, spw_index, channel_index, pol_index,
                                    1])
                                bv_weight[
                                    time_index, antenna2, antenna1, channel_no, pol_index] = _vis[
                                                                                             row,
                                                                                             :,
                                                                                             :,
                                                                                             spw_index,
                                                                                             channel_index,
                                                                                             pol_index,
                                                                                             2]
                        bv_uvw[time_index, antenna2, antenna1, 0] = uu[
                                                                        row] * constants.c.value
                        bv_uvw[time_index, antenna2, antenna1, 1] = vv[
                                                                        row] * constants.c.value
                        bv_uvw[time_index, antenna2, antenna1, 2] = ww[
                                                                        row] * constants.c.value
                        row += 1
            
            # Convert negative weights to flags
            bv_flags[bv_weight < 0.0] = 1
            bv_weight[bv_weight < 0.0] = 0.0
            
            vis_list.append(BlockVisibility(uvw=bv_uvw,
                                            time=bv_times,
                                            frequency=freq[spw_index][channum],
                                            channel_bandwidth=freq_delt[channum],
                                            vis=bv_vis, flags=bv_flags,
                                            weight=bv_weight,
                                            imaging_weight=bv_weight,
                                            configuration=configuration,
                                            phasecentre=phasecentre,
                                            polarisation_frame=polarisation_frame))
    return vis_list


def create_visibility_from_uvfits(fitsname, channum=None, ack=False, antnum=None):
    """ Minimal UVFITS to BlockVisibility converter

    Creates a list of BlockVisibility's, split by field and spectral window

    :param fitsname: File name of UVFITS file
    :param channum: range of channels e.g. range(17,32), default is None meaning all
    :param antnum: the number of antenna
    :return:
    """
    from rascil.processing_components.visibility.coalesce import \
        convert_blockvisibility_to_visibility
    return [convert_blockvisibility_to_visibility(v)
            for v in
            create_blockvisibility_from_uvfits(fitsname=fitsname, channum=channum,
                                               ack=ack, antnum=antnum)]


def calculate_visibility_phasor(direction, vis):
    """ Calculate the phasor for a direction for a Visibility

    :param direction:
    :param vis:
    :return:
    """
    l, m, n = skycoord_to_lmn(direction, vis.phasecentre)
    phasor = simulate_point(vis.uvw, l, m)[..., numpy.newaxis]
    return phasor


def calculate_blockvisibility_phasor(direction, vis):
    """ Calculate the phasor for a component for a BlockVisibility

    :param comp:
    :param vis:
    :return:
    """
    ntimes, nant, _, nchan, npol = vis.vis.shape
    k = numpy.array(vis.frequency) / constants.c.to('m s^-1').value
    l, m, n = skycoord_to_lmn(direction, vis.phasecentre)
    uvw = vis.uvw[..., numpy.newaxis] * k
    phasor = numpy.ones([ntimes, nant, nant, nchan, npol], dtype='complex')
    for chan in range(nchan):
        phasor[:, :, :, chan, :] = simulate_point(uvw[..., chan], l, m)[..., numpy.newaxis]
    return phasor
