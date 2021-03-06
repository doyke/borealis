=================
Config Parameters
=================

+---------------------------------------------+--------------------------------+--------------------------------------+
| Parameter                                   | Example                        | Description                          |
+=============================================+================================+======================================+
| site_id                                     | sas                            | 3-letter standard ID of the radar.   |
+---------------------------------------------+--------------------------------+--------------------------------------+
| devices                                     | addr0=192.168.10.100,          | Addresses of the USRPs.              |
|                                             | addr1=192.168.10.101,          |                                      |
|                                             | addr2=192.168.10.102,          |                                      |
|                                             | addr3=192.168.10.103           |                                      |
+---------------------------------------------+--------------------------------+--------------------------------------+
| main_antenna_count                          | 16                             | Number of main array antennas (TX)   |
+---------------------------------------------+--------------------------------+--------------------------------------+
| interferometer_antenna_count                | 4                              | Number of interferometer antennas    |
+---------------------------------------------+--------------------------------+--------------------------------------+
| main_antenna_spacing                        | 15.24                          | Distance between antennas (m)        |
+---------------------------------------------+--------------------------------+--------------------------------------+
| interferometer_antenna_spacing              | 15.24                          | Distance between antennas (m)        |
+---------------------------------------------+--------------------------------+--------------------------------------+
| min_freq                                    | 8.0e6                          | Minimum frequency we can run (Hz)    |
+---------------------------------------------+--------------------------------+--------------------------------------+
| max_freq                                    | 20.0e6                         | Maximum frequency we can run (Hz)    |
+---------------------------------------------+--------------------------------+--------------------------------------+
| minimum_pulse_length                        | 100                            | Minimum pulse length (us) dependent upon AGC feedback sample and hold           |
+---------------------------------------------+--------------------------------+--------------------------------------+
| minimum_mpinc_length                        | 1                              | Minimum length of multi-pulse        |
|                                             |                                | increment (us)                       |
+---------------------------------------------+--------------------------------+--------------------------------------+
| minimum_pulse_separation                    | 125                            | The minimum separation (us) before   |
|                                             |                                | experiment treats it as a single     |
|                                             |                                | pulse (transmitting zeroes and not   |
|                                             |                                | receiving between the pulses.        |
|                                             |                                | 125 us is approx two TX/RX times     |
+---------------------------------------------+--------------------------------+--------------------------------------+
| TODO : tx_subdev, main_rx_subdev??          |                                |                                      |
| TODO : interferometer_rx_subdev ??          |                                |                                      |
+---------------------------------------------+--------------------------------+--------------------------------------+
| tx_sample_rate                              | 10.0e6                         | Sampling rate (Hz) of TX O/P         |
+---------------------------------------------+--------------------------------+--------------------------------------+
| rx_sample_rate                              | 10.0e6                         | Sampling rate (Hz) of RX             |
+---------------------------------------------+--------------------------------+--------------------------------------+
| pps                                         | external                       | pulse per second source for timing   |
+---------------------------------------------+--------------------------------+--------------------------------------+
| ref                                         | external                       |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| overthewire                                 | sc16                           |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| cpu                                         | fc32                           |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| max_usrp_dac_amplitude                      | 0.7                            | Maximum amplitude to output given    |
|                                             |                                | DAC on TX board                      |
+---------------------------------------------+--------------------------------+--------------------------------------+
| pulse_ramp_time                             | 10.0e-6                        | Time for linear ramp-up for pulses   |
|                                             |                                | (s)                                  |
+---------------------------------------------+--------------------------------+--------------------------------------+
| gpio_bank                                   | RXA                            |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| scope_sync_mask                             | 1                              | Bit mapping for scope sync GPIO      |
+---------------------------------------------+--------------------------------+--------------------------------------+
| atten_mask                                  | 2                              | Bit mapping for attenuator GPIO      |
+---------------------------------------------+--------------------------------+--------------------------------------+
| tr_mask                                     | 4                              | Bit mapping for transmit/receive     |
|                                             |                                | GPIO                                 |
+---------------------------------------------+--------------------------------+--------------------------------------+
| atten_window_time_start                     | 10e-6                          |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| atten_window_time_end                       | 40e-6                          |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| tr_window_time                              | 60e-6                          |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| first_stage_sample_rate                     | 1.0-e6                         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| second_stage_sample_rate                    | 100.0e3                        |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| third_stage_sample_rate                     | 3.333e3                        |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| first_stage_filter_cutoff                   | 1.0e6                          |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| first_stage_filter_transition               | 500.0e3                        |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| second_stage_filter_cutoff                  | 100.0e3                        |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| second_stage_filter_transition              | 50.0e3                         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| third_stage_filter_cutoff                   | 3.333e3                        |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| third_stage_filter_transition               | 0.833e3                        |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| driver_to_rx_dsp_address                    | 'tcp://127.0.0.1:5000'         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| radar_control_to_rx_dsp_address             | 'tcp://127.0.0.1:5001'         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| radar_control_to_driver_address             | 'tcp://127.0.0.1:5002'         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| rx_dsp_to_radar_control_ack_address         | 'tcp://127.0.0.1:5003'         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| rx_dsp_to_radar_control_timing_address      | 'tcp://127.0.0.1:5004'         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| experiment_handler_to_radar_control_address | 'tcp://127.0.0.1:5005'         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
| data_to_experiment_address                  | 'tcp://127.0.0.1:5006'         |    ??                                |
+---------------------------------------------+--------------------------------+--------------------------------------+
