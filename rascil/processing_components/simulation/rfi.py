"""Functions used to simulate RFI. Developed as part of SP-122/SIM.

The scenario is:
* There is a TV station at a remote location (e.g. Perth), emitting a broadband signal (7MHz) of known power (50kW).
* The emission from the TV station arrives at LOW stations with phase delay and attenuation. Neither of these are
well known but they are probably static.
* The RFI enters LOW stations in a side-lobe of the main beam. Calculations by Fred Dulwich indicate that this
provides attenuation of about 55 - 60dB for a source close to the horizon.
* The RFI enters each LOW station with fixed delay and zero fringe rate (assuming no e.g. ionospheric ducting)
* In tracking a source on the sky, the signal from one station is delayed and fringe-rotated to stop the fringes for
one direction on the sky.
* The fringe rotation stops the fringe from a source at the phase tracking centre but phase rotates the RFI, which
now becomes time-variable.
* The correlation data are time- and frequency-averaged over a timescale appropriate for the station field of view.
This averaging de-correlates the RFI signal.
* We want to study the effects of this RFI on statistics of the images: on source and at the pole.
"""

__all__ = ['simulate_DTV', 'simulate_DTV_prop', 'create_propagators', 'create_propagators_prop',
           'calculate_averaged_correlation', 'calculate_rfi_at_station',
           'simulate_rfi_block', 'simulate_rfi_block_prop', 'calculate_station_correlation_rfi']

import astropy.units as u
import numpy
import copy
from astropy import constants
from astropy.coordinates import SkyCoord, EarthLocation

from rascil.processing_components.util.array_functions import average_chunks2
from rascil.processing_components.util.compass_bearing import calculate_initial_compass_bearing
from rascil.processing_components.util.coordinate_support import simulate_point
from rascil.processing_components.util.coordinate_support import skycoord_to_lmn, azel_to_hadec


def simulate_DTV_prop(frequency, times, power=50e3, freq_cen=177.5e06, bw=7e06, timevariable=False,
                      frequency_variable=False):
    """ Calculate DTV sqrt(power) as a function of time and frequency

    :param frequency: (sample frequencies)
    :param times: sample times (s)
    :param power: DTV emitted power W
    :param freq_cen: central frequency of DTV
    :parm bw: bandwidth of DTV
    :param timevariable
    :param frequency_variable
    :return: Complex array [ntimes, nchan]
    """
    # find frequency range for DTV
    DTVrange = numpy.where((frequency <= freq_cen + (bw / 2.)) & (frequency >= freq_cen - (bw / 2.)))
    # idx = (np.abs(frequency - freq_cen)).argmin()
    # print(frequency, freq_cen, bw, DTVrange)
    echan = DTVrange[0].max()
    bchan = DTVrange[0].min()
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    # bchan = nchan // 4
    # echan = 3 * nchan // 4
    amp = numpy.sqrt(power / (frequency[echan] - frequency[bchan]))
    # print(frequency[bchan], frequency[echan], frequency)
    # print('DTV', DTVrange, bchan, echan, amp)

    signal = numpy.zeros(shape, dtype='complex')
    if timevariable:
        if frequency_variable:
            sshape = [ntimes, nchan // 2]
            signal[:, bchan:echan + 1] += numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape) \
                                          + 1j * numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape)
        else:
            sshape = [ntimes]
            signal[:, bchan:echan + 1] += numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape) \
                                          + 1j * numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape)
    else:
        if frequency_variable:
            sshape = [nchan // 2]
            signal[:, bchan:echan + 1] += (numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape)
                                           + 1j * numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape))[
                numpy.newaxis, ...]
        else:
            signal[:, bchan:echan + 1] = amp
    return signal, [bchan, echan]


def simulate_DTV(frequency, times, power=50e3, timevariable=False, frequency_variable=False):
    """ Calculate DTV sqrt(power) as a function of time and frequency
    :param frequency: (sample frequencies)
    :param times: sample times (s)
    :param power: DTV emitted power W
    :param frequency_variable:
    :param timevariable:
    :return: Complex array [ntimes, nchan]
    """
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    bchan = nchan // 4
    echan = 3 * nchan // 4
    amp = numpy.sqrt(power / (max(frequency) - min(frequency)))
    signal = numpy.zeros(shape, dtype='complex')
    if timevariable:
        if frequency_variable:
            sshape = [ntimes, nchan // 2]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape) \
                                      + 1j * numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape)
        else:
            sshape = [ntimes]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape) \
                                      + 1j * numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape)
    else:
        if frequency_variable:
            sshape = [nchan // 2]
            signal[:, bchan:echan] += (numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape)
                                       + 1j * numpy.random.normal(0.0, numpy.sqrt(amp / 2.0), sshape))[
                numpy.newaxis, ...]
        else:
            signal[:, bchan:echan] = amp

    return signal


def create_propagators(config, interferer, frequency, attenuation=1e-9):
    """ Create a set of propagators
    :return: Complex array [nants, ntimes]
    """
    nchannels = len(frequency)
    nants = len(config.data['names'])
    interferer_xyz = [interferer.geocentric[0].value, interferer.geocentric[1].value, interferer.geocentric[2].value]
    propagators = numpy.zeros([nants, nchannels], dtype='complex')
    for iant, ant_xyz in enumerate(config.xyz):
        vec = ant_xyz - interferer_xyz
        # This ignores the Earth!
        r = numpy.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        k = 2.0 * numpy.pi * frequency / constants.c.value
        propagators[iant, :] = numpy.exp(- 1.0j * k * r) / r
    return propagators * attenuation


def create_propagators_prop(config, frequency, nants_start, station_skip=1, attenuation=1e-9, beamgainval=0.,
                            trans_range=[0, None]):
    """ Create a set of propagators
    :param config: configuration
    :param frequency: frequencies
    :param transmitter: str name of transmitter
    :param attenuation: generic attenuation value to use if no transmitter specified, else filename to load
    :param beamgainval: float generic beam gain value to use if no transmitter specified, else filename to load
    :param nants_start: limiting station to use determined by use of rmax
    :param station_skip: number of stations to skip
    :param trans_range: array start and stop channels for applying the attenuation and beam gain
    :return: Complex array [nants, ntimes]
    """

    nchannels = len(frequency)
    nants = len(config.data['names'])
    propagation = numpy.ones([nants, nchannels], dtype='complex')
    if isinstance(attenuation, str):
        propagation_trans = numpy.power(10, -1 * numpy.load(attenuation) / 10.)
    else:
        propagation_trans = attenuation
    if isinstance(beamgainval, str):
        beamgain_trans = numpy.loadtxt(beamgainval)
    else:
        beamgain_trans = beamgainval
    propagation_trans *= beamgain_trans
    if type(propagation_trans) is numpy.ndarray:
        propagation_trans = propagation_trans[:nants_start]
        propagation_trans = propagation_trans[::station_skip]
    if trans_range[1] is None:
        propagation[:, trans_range[0]:trans_range[1]] = propagation_trans
    else:
        propagation[:, trans_range[0]:trans_range[1] + 1] = propagation_trans

    '''
    if transmitter:
        # propagation = numpy.power(10, -1 * numpy.load(direct + 'Attenuation_' + transmitter + '.npy') / 10.)
        propagation = numpy.power(10, -1 * numpy.load(attenuation) / 10.)
        #beamgain = numpy.loadtxt(bg_direct + 'Beam_gain_' + beam_file + '.txt')
        beam_gain = numpy.loadtxt(beamgain)
        # trans_position = gain_list.index(transmitter)
        # propagation[:, :] *= beamgain[trans_position]
        propagation[:, :] *= beam_gain
        propagation = propagation[:nants_start]
        propagation = propagation[::station_skip]
    else:
        nants = len(config.data['names'])
        propagation = numpy.ones([nants, nchannels], dtype='complex')
        propagation[:, :] = attenuation
    '''

    propagation = numpy.sqrt(propagation)
    # print(propagation.type)
    return propagation


def calculate_rfi_at_station(propagators, emitter):
    """ Calculate the rfi at each station
    :param propagators: [nstations, nchannels]
    :param emitter: [ntimes, nchannels]
    :return: Complex array [nstations, ntimes, nchannels]
    """
    rfi_at_station = emitter[:, numpy.newaxis, ...] * propagators[numpy.newaxis, ...]
    # rfi_at_station[numpy.abs(rfi_at_station)<1e-15] = 0.
    return rfi_at_station


def calculate_station_correlation_rfi(rfi_at_station):
    """ Form the correlation from the rfi at the station
    :param rfi_at_station:
    :return: Correlation(nant, nants, ntimes, nchan] in Jy
    """
    ntimes, nants, nchan = rfi_at_station.shape
    correlation = numpy.zeros([ntimes, nants, nants, nchan], dtype='complex')

    for itime in range(ntimes):
        for chan in range(nchan):
            correlation[itime, ..., chan] = numpy.outer(rfi_at_station[itime, :, chan],
                                                        numpy.conjugate(rfi_at_station[itime, :, chan]))
    return correlation[..., numpy.newaxis] * 1e26


def calculate_averaged_correlation(correlation, time_width, channel_width):
    """ Average the correlation in time and frequency
    :param correlation: Correlation(nant, nants, ntimes, nchan]
    :param channel_width: Number of channels to average
    :param time_width: Number of integrations to average
    :return:
    """
    wts = numpy.ones(correlation.shape, dtype='float')
    return average_chunks2(correlation, wts, (time_width, channel_width))[0]


def simulate_rfi_block(bvis, emitter_location, emitter_power=5e4, attenuation=1.0, use_pole=False):
    """ Simulate RFI block
    :param emitter_location: EarthLocation of emitter
    :param emitter_power: Power of emitter
    :param attenuation: Attenuation to be applied to signal
    :param use_pole: Set the emitter to nbe at the southern celestial pole
    :return:
    """

    # Calculate the power spectral density of the DTV station: Watts/Hz
    emitter = simulate_DTV(bvis.frequency, bvis.time, power=emitter_power, timevariable=False)

    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency. The ad hoc attenuation
    # is set to produce signal roughly equal to noise at LOW
    propagators = create_propagators(bvis.configuration, emitter_location, frequency=bvis.frequency,
                                     attenuation=attenuation)
    # Now calculate the RFI at the stations, based on the emitter and the propagators
    rfi_at_station = calculate_rfi_at_station(propagators, emitter)

    # Calculate the rfi correlation using the fringe rotation and the rfi at the station
    # [ntimes, nants, nants, nchan, npol]
    bvis.data['vis'][...] = calculate_station_correlation_rfi(rfi_at_station)

    ntimes, nant, _, nchan, npol = bvis.vis.shape

    s2r = numpy.pi / 43200.0
    k = numpy.array(bvis.frequency) / constants.c.to('m s^-1').value
    uvw = bvis.uvw[..., numpy.newaxis] * k

    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')

    if use_pole:
        # Calculate phasor needed to shift from the phasecentre to the pole
        l, m, n = skycoord_to_lmn(pole, bvis.phasecentre)
        phasor = numpy.ones([ntimes, nant, nant, nchan, npol], dtype='complex')
        for chan in range(nchan):
            phasor[:, :, :, chan, :] = simulate_point(uvw[..., chan], l, m)[..., numpy.newaxis]

        # Now fill this into the BlockVisibility
        bvis.data['vis'] = bvis.data['vis'] * phasor
    else:
        # We know where the emitter is. Calculate the bearing to the emitter from
        # the site, generate az, el, and convert to ha, dec. ha, dec is static.
        site = bvis.configuration.location
        site_tup = (site.lat.deg, site.lon.deg)
        emitter_tup = (emitter_location.lat.deg, emitter_location.lon.deg)
        az = - calculate_initial_compass_bearing(site_tup, emitter_tup) * numpy.pi / 180.0
        el = 0.0
        hadec = azel_to_hadec(az, el, site.lat.rad)

        # Now step through the time stamps, calculating the effective
        # sky position for the emitter, and performing phase rotation
        # appropriately
        for itime, time in enumerate(bvis.time):
            ra = - hadec[0] + s2r * time
            dec = hadec[1]
            emitter_sky = SkyCoord(ra * u.rad, dec * u.rad)
            l, m, n = skycoord_to_lmn(emitter_sky, bvis.phasecentre)

            phasor = numpy.ones([nant, nant, nchan, npol], dtype='complex')
            for chan in range(nchan):
                phasor[:, :, chan, :] = simulate_point(uvw[itime, ..., chan], l, m)[..., numpy.newaxis]

            # Now fill this into the BlockVisibility
            bvis.data['vis'][itime, ...] = bvis.data['vis'][itime, ...] * phasor

    return bvis


def check_prop_parms(attenuation_state, beamgain_state, transmitter_list):
    if attenuation_state is None:
        attenuation_value = 1.0
        att_context = 'att_value'
    else:
        attenuation_value = attenuation_state[0]
        att_context = attenuation_state[1]
    if beamgain_state is None:
        beamgain_value = 1.0
        bg_context = 'bg_value'
    else:
        beamgain_value = beamgain_state[0]
        bg_context = beamgain_state[1]
    if not transmitter_list:
        transmitter_list = {'Test_DTV': {'location': [115.8605, -31.9505], 'power': 50000.0, 'height': 175},
                            'freq': 177.5, 'bw': 7}

    return attenuation_value, att_context, beamgain_value, bg_context, transmitter_list


def get_file_strings(attenuation_value, att_context, beamgain_value, bg_context, trans):
    if att_context == 'att_dir':
        attenuation = attenuation_value + 'Attenuation_' + trans + '.npy'
    elif att_context == 'att_file':
        attenuation = attenuation_value

    if bg_context == 'bg_dir':
        beamgain = beamgain_value + trans + '_beam_gain_TIME_SEP_CHAN_SEP_CROSS_POWER_AMP_I_I.txt'
    elif bg_context == 'bg_file':
        beamgain = beamgain_value

    return attenuation, beamgain


def simulate_rfi_block_prop(bvis, nants_start, station_skip, attenuation_state=None,
                            beamgain_state=None, use_pole=False, transmitter_list=None):
    """ Simulate RFI block
    :param transmitter_list: dictionary of transmitters
    :param beamgain_state: beam gains to apply to the signal or file containing values and flag to declare which
    :param attenuation_state: Attenuation to be applied to signal or file containing values and flag to declare which
    :param use_pole: Set the emitter to nbe at the southern celestial pole
    :return:
    """

    # sort inputs
    attenuation_value, att_context, beamgain_value, bg_context, transmitter_list = check_prop_parms(attenuation_state,
                                                                                                    beamgain_state,
                                                                                                    transmitter_list)

    # temporary copy to calculate contribution for each transmitter
    bvis_data_copy = copy.copy(bvis.data)

    for trans in transmitter_list:

        print('Processing transmitter', trans)
        emitter_power = transmitter_list[trans]['power']
        emitter_location = EarthLocation(lon=transmitter_list[trans]['location'][0],
                                         lat=transmitter_list[trans]['location'][1],
                                         height=transmitter_list[trans]['height'])
        emitter_freq = transmitter_list[trans]['freq'] * 1e06
        emitter_bw = transmitter_list[trans]['bw'] * 1e06

        attenuation, beamgain = get_file_strings(attenuation_value, att_context, beamgain_value, bg_context, trans)

        # Calculate the power spectral density of the DTV station: Watts/Hz
        emitter, DTV_range = simulate_DTV_prop(bvis.frequency, bvis.time, power=emitter_power, freq_cen=emitter_freq,
                                               bw=emitter_bw, timevariable=False)

        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The ad hoc attenuation
        # is set to produce signal roughly equal to noise at LOW
        propagators = create_propagators_prop(bvis.configuration, bvis.frequency, nants_start=nants_start,
                                              station_skip=station_skip, attenuation=attenuation,
                                              transmitter=trans, beamgainval=beamgain, trans_range=DTV_range)
        # Now calculate the RFI at the stations, based on the emitter and the propagators
        rfi_at_station = calculate_rfi_at_station(propagators, emitter)

        # Calculate the rfi correlation using the fringe rotation and the rfi at the station
        # [ntimes, nants, nants, nchan, npol]

        # bvis.data['vis'][...] += calculate_station_correlation_rfi(rfi_at_station)
        bvis_data_copy['vis'][...] = calculate_station_correlation_rfi(rfi_at_station)

        ntimes, nant, _, nchan, npol = bvis.vis.shape

        s2r = numpy.pi / 43200.0
        k = numpy.array(bvis.frequency) / constants.c.to('m s^-1').value
        uvw = bvis.uvw[..., numpy.newaxis] * k

        pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')

        if use_pole:
            # Calculate phasor needed to shift from the phasecentre to the pole
            l, m, n = skycoord_to_lmn(pole, bvis.phasecentre)
            phasor = numpy.ones([ntimes, nant, nant, nchan, npol], dtype='complex')
            for chan in range(nchan):
                phasor[:, :, :, chan, :] = simulate_point(uvw[..., chan], l, m)[..., numpy.newaxis]

            # Now fill this into the BlockVisibility
            # bvis.data['vis'] = bvis.data['vis'] * phasor
            bvis.data['vis'] += bvis_data_copy['vis'] * phasor
        else:
            # We know where the emitter is. Calculate the bearing to the emitter from
            # the site, generate az, el, and convert to ha, dec. ha, dec is static.
            site = bvis.configuration.location
            site_tup = (site.lat.deg, site.lon.deg)
            emitter_tup = (emitter_location.lat.deg, emitter_location.lon.deg)
            az = - calculate_initial_compass_bearing(site_tup, emitter_tup) * numpy.pi / 180.0
            el = 0.0
            hadec = azel_to_hadec(az, el, site.lat.rad)

            # print(trans, 'bvis_data', bvis.data['vis'][0][0][0])
            # print(trans, 'bvis_copy_data', bvis_data_copy['vis'][0][0][0])

            # Now step through the time stamps, calculating the effective
            # sky position for the emitter, and performing phase rotation
            # appropriately
            for itime, time in enumerate(bvis.time):
                ra = - hadec[0] + s2r * time
                dec = hadec[1]
                emitter_sky = SkyCoord(ra * u.rad, dec * u.rad)
                l, m, n = skycoord_to_lmn(emitter_sky, bvis.phasecentre)

                phasor = numpy.ones([nant, nant, nchan, npol], dtype='complex')
                for chan in range(nchan):
                    phasor[:, :, chan, :] = simulate_point(uvw[itime, ..., chan], l, m)[..., numpy.newaxis]

                # Now fill this into the BlockVisibility
                # bvis.data['vis'][itime, ...] = bvis.data['vis'][itime, ...] * phasor
                bvis.data['vis'][itime, ...] += bvis_data_copy['vis'][itime, ...] * phasor

            # print(trans, 'bvis_data_new', bvis.data['vis'][0][0][0])

    return bvis
