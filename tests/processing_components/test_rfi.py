""" Unit tests for RFI simulation

"""

import logging
import unittest

import astropy.units as u
import numpy
import numpy.testing
from astropy.coordinates import SkyCoord, EarthLocation

from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.simulation.rfi import create_propagators, calculate_rfi_at_station, \
    calculate_station_correlation_rfi, simulate_DTV, simulate_DTV_prop, create_propagators_prop

log = logging.getLogger(__name__)


class TestRFISim(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_rfi(self):
        sample_freq = 3e4
        nchannels = 1000
        frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
        
        ntimes = 100
        integration_time = 0.5
        times = numpy.arange(ntimes) * integration_time
        
        # Perth from Google for the moment
        perth = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
        
        rmax = 1000.0
        low = create_named_configuration('LOWR3', rmax=rmax)
        antskip = 33
        low.data = low.data[::antskip]
        nants = len(low.names)
        
        # Calculate the power spectral density of the DTV station: Watts/Hz
        emitter = simulate_DTV(frequency, times, power=50e3, timevariable=False)
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.0408452567)
        assert emitter.shape == (ntimes, nchannels)
        
        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The ad hoc attenuation
        # is set to produce signal roughly equal to noise at LOW
        attenuation = 1.0
        propagators = create_propagators(low, perth, frequency=frequency, attenuation=attenuation)
        assert propagators.shape == (nants, nchannels), propagators.shape
        
        # Now calculate the RFI at the stations, based on the emitter and the propagators
        rfi_at_station = calculate_rfi_at_station(propagators, emitter)
        assert rfi_at_station.shape == (ntimes, nants, nchannels), rfi_at_station.shape
        
        # Calculate the rfi correlation
        # [nants, nants, ntimes, nchan]
        correlation = calculate_station_correlation_rfi(rfi_at_station)
        assert correlation.shape == (ntimes, nants, nants, nchannels, 1), correlation.shape


    def test_rfi_prop(self):
        sample_freq = 3e4
        nchannels = 1000
        frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq

        ntimes = 100
        integration_time = 0.5
        times = numpy.arange(ntimes) * integration_time

        rmax = 1000.0
        low = create_named_configuration('LOWR3', rmax=rmax)
        nants_start = len(low.names)
        antskip = 33
        low.data = low.data[::antskip]
        nants = len(low.names)

        # Perth transmitter
        tx_name = 'Perth'
        transmitter_dict = {'location': [115.8605, -31.9505], 'power': 50000.0, 'height': 175, 'freq': 177.5,
                            'bw': 7}

        # Calculate the power spectral density of the DTV station using the transmitter bandwidth and frequency
        # : Watts/Hz
        emitter, channel_range = simulate_DTV_prop(frequency, times, power=transmitter_dict['power'],
                                                   freq_cen=transmitter_dict['freq']*1e6, bw=transmitter_dict['bw']*1e6,
                                                   timevariable=False, frequency_variable=False)
        numpy.testing.assert_almost_equal(numpy.max(numpy.abs(emitter)), 0.084575858)
        assert len(channel_range) == 2
        assert channel_range[0] == 117
        assert channel_range[1] == 350
        assert emitter.shape == (ntimes, nchannels)

        # Calculate the propagators for signals from Perth to the stations in low
        # These are fixed in time but vary with frequency. The attenuation and beam gain
        # are set per frequency channel covered by the transmitter bandwidth.

        propagators = create_propagators_prop(low, frequency, nants_start, station_skip=antskip, attenuation=1e-9,
                                              beamgainval=1e-8, trans_range=channel_range)
        assert propagators.shape == (nants, nchannels), propagators.shape






if __name__ == '__main__':
    unittest.main()
