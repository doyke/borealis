#!/usr/bin/env python

# Copyright 2017 SuperDARN Canada

"""
To load the config options to be used by the experiment and radar_control blocks. 
Config data comes from the config.ini file and the hdw.dat file.
"""

import json
import datetime
import os

from experiments.experiment_exception import ExperimentException

borealis_path = os.environ['BOREALISPATH']
config_file = borealis_path + '/config.ini'
hdw_dat_file = borealis_path + '/hdw.dat.'
restricted_freq_file = borealis_path + '/restrict.dat.'

class ExperimentOptions:
    # TODO long init file, consider using multiple functions
    def __init__(self):
        """
        Create an object of necessary hardware and site parameters for use in the experiment.
        """
        try:
            with open(config_file) as config_data:
                config = json.load(config_data)
        except IOError:
            errmsg = 'Cannot open config file at {}'.format(config_file)
            raise ExperimentException(errmsg)

        try:
            self.main_antenna_count = int(config['main_antenna_count'])
            self.interferometer_antenna_count = int(config['interferometer_antenna_count'])
            self.main_antenna_spacing = float(config['main_antenna_spacing'])
            self.interferometer_antenna_spacing = float(config['interferometer_antenna_spacing'])
            self.tx_sample_rate = float(config['tx_sample_rate'])
            self.rx_sample_rate = float(config['rx_sample_rate'])
            self.max_usrp_dac_amplitude = float(config['max_usrp_dac_amplitude'])
            self.pulse_ramp_time = float(config['pulse_ramp_time'])  # in seconds
            self.tr_window_time = float(config['tr_window_time'])
            self.output_sample_rate = float(
                config['third_stage_sample_rate'])  # should use to check iqdata samples
            # when adjusting the experiment during operations.
            self.filter_description = {'filter_cutoff': config['third_stage_filter_cutoff'],
                                       'filter_transition': config['third_stage_filter_transition']}
            self.site_id = config['site_id']
            self.max_freq = float(config['max_freq'])  # Hz
            self.min_freq = float(config['min_freq'])  # Hz
            self.minimum_pulse_length = float(config['minimum_pulse_length'])  # us
            self.minimum_mpinc_length = float(config['minimum_mpinc_length'])  # us
            # Minimum pulse separation is the minimum before the experiment treats it as a single pulse (transmitting zeroes or no receiving between the pulses)
            # 125 us is approx two TX/RX times
            self.minimum_pulse_separation = float(config['minimum_pulse_separation'])  # us
            self.tr_window_time = float(config['tr_window_time'])  # s
            self.atten_window_time_start = float(config['atten_window_time_start'])  # s
            self.atten_window_time_end = float(config['atten_window_time_end'])  # s
            self.experiment_handler_to_radar_control_address = config[
                'experiment_handler_to_radar_control_address']
            self.data_to_experiment_address = config['data_to_experiment_address']
            self.radar_control_to_driver_address = config['radar_control_to_driver_address']
            self.radar_control_to_rx_dsp_address = config['radar_control_to_rx_dsp_address']
            self.rx_dsp_to_radar_control_ack_address = config['rx_dsp_to_radar_control_ack_address']
            self.rx_dsp_to_radar_control_timing_address = \
                config['rx_dsp_to_radar_control_timing_address']
            # TODO add appropriate signal process maximum time here after timing is changed - can use to check for pulse spacing minimums, pace the driver
        except ValueError as e:
            # TODO: error
            raise e

        today = datetime.datetime.today()
        year_start = datetime.datetime(today.year, 1, 1, 0, 0, 0, 0)  # start of the year
        year_timedelta = today - year_start

        try:
            with open(hdw_dat_file + self.site_id) as hdwdata:
                lines = hdwdata.readlines()
        except IOError:
            errmsg = 'Cannot open hdw.dat.{} file at {}'.format(self.site_id, (hdw_dat_file + self.site_id))
            raise ExperimentException(errmsg)

        lines[:] = [line for line in lines if line[0] != "#"]  # remove comments
        lines[:] = [line for line in lines if len(line.split()) != 0]  # remove blanks
        lines[:] = [line for line in lines if int(line.split()[1]) > today.year or
                    (int(line.split()[1]) == today.year and float(line.split()[2]) >
                     year_timedelta.total_seconds())]  # only take present & future hdw data

        # there should only be one line left, however if there are more we will take the
        # one that expires first.
        if len(lines) > 1:
            times = [[line.split()[1], line.split()[2]] for line in lines]
            min_year = times[0][0]
            min_yrsec = times[0][1]
            hdw_index = 0
            for i in range(len(times)):
                year = times[i][0]
                yrsec = times[i][1]
                if year < min_year:
                    hdw_index = i
                elif year == min_year:
                    if yrsec < min_yrsec:
                        hdw_index = i
            hdw = lines[hdw_index]
        else:
            try:
                hdw = lines[0]
            except IndexError:
                errmsg = 'Cannot find any valid lines for this time period in the hardware file ' \
                         '{}'.format((hdw_dat_file + self.site_id))
                raise ExperimentException(errmsg)
        # we now have the correct line of data.

        params = hdw.split()
        if len(params) != 19:
            errmsg = 'Found {} parameters in hardware file, expected 19'.format(len(params))
            raise ExperimentException(errmsg)

        self.geo_lat = params[3]  # decimal degrees, S = negative
        self.geo_long = params[4]  # decimal degrees, W = negative
        self.altitude = params[5]  # metres
        self.boresight = params[6]  # degrees from geographic north, CCW = negative.
        self.beam_sep = params[7]  # degrees TODO is this necessary, or is this a min. - for post-processing software in RST? check with others.
        self.velocity_sign = params[8]  # +1.0 or -1.0
        self.analog_rx_attenuator = params[9]  # dB
        self.tdiff = params[10] # ns
        self.phase_sign = params[11]
        self.intf_offset = [float(params[12]), float(params[13]), float(params[14])]  # interferometer offset from
        # midpoint of main, metres [x, y, z] where x is along line of antennas, y is along array
        # normal and z is altitude difference.
        self.analog_rx_rise = params[15]  # us
        self.analog_atten_stages = params[16]  # number of stages
        self.max_range_gates = params[17]
        self.max_beams = params[18]  # so a beam number always points in a certain direction
                        # TODO Is this last one necessary - why don't we specify directions in angle. - also for post-processing so check if it applies to Borealis

        try:
            with open(restricted_freq_file + self.site_id) as restricted_freq_data:
                restricted = restricted_freq_data.readlines()
        except IOError:
            errmsg = 'Cannot open restrict.dat.{} file at {}'.format(self.site_id,
                                                                (restricted_freq_file + self.site_id))
            raise ExperimentException(errmsg)

        restricted[:] = [line for line in restricted if line[0] != "#"]  # remove comments
        restricted[:] = [line for line in restricted if len(line.split()) != 0]  # remove blanks

        for line in restricted:
            splitup = line.split("=")
            if len(splitup) == 2:
                if splitup[0] == 'default' or splitup[0] == 'default ':
                    self.default_freq = int(splitup[1]) # kHz
                    restricted.remove(line)
                    break
        else: #no break
            raise Exception('No Default Frequency Found in Restrict.dat')

        self.restricted_ranges = []
        for line in restricted:
            splitup = line.split()
            if len(splitup) != 2:
                raise Exception('Problem with Restricted Frequency: A Range Len != 2')
            try:
                splitup = [int(float(freq)) for freq in splitup]  # convert to ints
            except ValueError:
                raise ValueError('Error parsing Restrict.Dat Frequency Ranges, Invalid Literal')
            restricted_range = tuple(splitup)
            self.restricted_ranges.append(restricted_range)

    def __repr__(self):
        return_str = """\n    main_antenna_count = {} \
                    \n    interferometer_antenna_count = {} \
                    \n    main_antenna_spacing = {} metres \
                    \n    interferometer_antenna_spacing = {} metres \
                    \n    tx_sample_rate = {} Hz (samples/sec)\
                    \n    rx_sample_rate = {} Hz (samples/sec)\
                    \n    max_usrp_dac_amplitude = {} : 1\
                    \n    pulse_ramp_time = {} s\
                    \n    tr_window_time = {} s\
                    \n    output_sample_rate = {} Hz\
                    \n    filter_description = {} \
                    \n    site_id = {} \
                    \n    geo_lat = {} degrees \
                    \n    geo_long = {} degrees\
                    \n    altitude = {} metres \
                    \n    boresight = {} degrees from geographic north, CCW = negative. \
                    \n    beam_sep = {} degrees\
                    \n    velocity_sign = {} \
                    \n    analog_rx_attenuator = {} dB \
                    \n    tdiff = {} us \
                    \n    phase_sign = {} \
                    \n    intf_offset = {} \
                    \n    analog_rx_rise = {} us \
                    \n    analog_atten_stages = {} \
                    \n    max_range_gates = {} \
                    \n    max_beams = {} \
                    \n    max_freq = {} \
                    \n    min_freq = {} \
                    \n    minimum_pulse_length = {} \
                    \n    minimum_mpinc_length = {} \
                    \n    minimum_pulse_separation = {} \
                    \n    tr_window_time = {} \
                    \n    atten_window_time_start = {} \
                    \n    atten_window_time_end = {} \
                    \n    default_freq = {} \
                    \n    restricted_ranges = {} \
                    \n    experiment_handler_to_radar_control_address = {} \
                    \n    data_to_experiment_address = {} \
                    \n    radar_control_to_rx_dsp_address = {} \
                    \n    radar_control_to_driver_address = {} \
                    \n    rx_dsp_to_radar_control_ack_address = {} \
                    \n    rx_dsp_to_radar_control_timing_address = {} \
                     """.format(self.main_antenna_count, self.interferometer_antenna_count,
                                self.main_antenna_spacing, self.interferometer_antenna_spacing,
                                self.tx_sample_rate, self.rx_sample_rate,
                                self.max_usrp_dac_amplitude, self.pulse_ramp_time,
                                self.tr_window_time, self.output_sample_rate,
                                self.filter_description, self.site_id, self.geo_lat, self.geo_long,
                                self.altitude, self.boresight, self.beam_sep, self.velocity_sign,
                                self.analog_rx_attenuator, self.tdiff, self.phase_sign,
                                self.intf_offset, self.analog_rx_rise, self.analog_atten_stages,
                                self.max_range_gates, self.max_beams, self.max_freq, self.min_freq,
                                self. minimum_pulse_length, self.minimum_mpinc_length,
                                self.minimum_pulse_separation, self.tr_window_time,
                                self.atten_window_time_start, self.atten_window_time_end,
                                self.default_freq, self.restricted_ranges,
                                self.experiment_handler_to_radar_control_address,
                                self.data_to_experiment_address,
                                self.radar_control_to_rx_dsp_address,
                                self.radar_control_to_driver_address,
                                self.rx_dsp_to_radar_control_ack_address,
                                self.rx_dsp_to_radar_control_timing_address)
        return return_str

