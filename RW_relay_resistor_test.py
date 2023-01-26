# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:55:06 2022

@author: UWTUCANMag
"""

from nidaqmx.constants import LineGrouping
import nidaqmx
import time
import os
from nidaqmx import stream_writers
from nidaqmx import stream_readers
from datetime import datetime
from tqdm import tqdm
import nidaqmx as ni
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')


# set up graphs
plt.ion()
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.bottom'] = True
#mpl.rcParams['agg.path.chunksize'] = 200

"""
Relay class values definition
"""

value = 1

notvalue = ~value & 0xff  
# turn off all switches

value = 0
notvalue = ~value & 0xff

# turn on the second switch

value = value | (1 << 2)
notvalue = ~value & 0xff

# now also turn on the fourth switch

value = value | (1 << 4)
notvalue = ~value & 0xff

# If I do the same thing again, nothing bad happens; switch 5 stays on

value = value | (1 << 4)
notvalue = ~value & 0xff

# now turn off the third switch

value = value & (~(1 << 2))
notvalue = ~value & 0xff

# If I do the same thing again, nothing bad happens, switch 3 stays off

value = value & (~(1 << 2))
notvalue = ~value & 0xff


class relay():

    pt = 2
    pt_2 = 2/100
    dev_str = "Dev1/port0/line0:7"

    def __init__(self, chan=0, value=0):
        self.value = value
        self.chan = chan

    def turn_on(self, switch):
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(self.dev_str)
        self.value = self.value | (1 << switch)
        task.write(self.value)
        task.close()
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(
            self.dev_str, line_grouping=LineGrouping.CHAN_PER_LINE)
        # readback do values
        print(task.read(number_of_samples_per_channel=1))
        time.sleep(self.pt_2)
        task.close()

    def turn_off(self, switch):
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(self.dev_str)
        self.value = self.value & (~(1 << switch))
        task.write(self.value)
        task.close()
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(
            self.dev_str, line_grouping=LineGrouping.CHAN_PER_LINE)
        # readback do values
        print(task.read(number_of_samples_per_channel=1))
        time.sleep(self.pt_2)
        task.close()

    def all_off(self):
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(self.dev_str)
        self.value = 0
        task.write(self.value)
        task.close()
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(
            self.dev_str, line_grouping=LineGrouping.CHAN_PER_LINE)
        # readback do values
        print(task.read(number_of_samples_per_channel=1))
        time.sleep(self.pt_2)
        task.close()

    def all_on(self):
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(self.dev_str)
        self.value = 0xff
        task.write(self.value)
        task.close()
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(
            self.dev_str, line_grouping=LineGrouping.CHAN_PER_LINE)
        # readback do values
        print(task.read(number_of_samples_per_channel=1))
        time.sleep(self.pt_2)
        task.close()

    def get_switches(self):
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(self.dev_str)
        notvalue = ~self.value & 0xff
        task.write(self.value)
        task.close()
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(
            self.dev_str, line_grouping=LineGrouping.CHAN_PER_LINE)
        # readback do values
        print(task.read(number_of_samples_per_channel=1))
        time.sleep(self.pt_2)
        task.close()
        return(notvalue)

    def set_switches(self, value=0):
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(self.dev_str)
        task.close()
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(
            self.dev_str, line_grouping=LineGrouping.CHAN_PER_LINE)
        # readback do values
        print(task.read(number_of_samples_per_channel=1))
        time.sleep(self.pt_2)
        task.close()
        self.value = value


class NIDAQ_RW(object):

    # device path
    device_name = 'Dev1'

    # signal out channels
    ao = ['/ao0',  # reference coil
          ]

    # readback channels and titles
    ai = [('/ai0', 'Resistor s1 (V)'),
          ('/ai1', 'Resistor s2 (V)')
          ]

    # arguments to pass to add_ao_voltage_chan
    ao_args = {'min_val': -10,
               'max_val': 10}

    ai_args = {'min_val': -10,
               'max_val': 10}

    # experiment details
    exp = ''
    title = ''
    position = ''

    # set the number of frames in a buffer (override)
    # the data gets written in chunks, each chunk is a frame
    # NOTE  With my NI6211 it was necessary to override the default buffer
    # size to prevent under/over run at high sample rates
    _frames_per_buffer = 10

    def __init__(self, readback_freq=2e4,
                 samples_per_channel=1e4,
                 terminal_config='RSE',
                 exp=None,
                 title=None,
                 position=None,
                 **settings):
        """
            readback_freq:          in Hz, rate of data taking and output. 
                                    Needs to be sufficiently high to prevent write errors
                                        1000 is too low
                                        100000 is good

            samples_per_channel:    set buffer size. If you specify 
                                    samples per channel of 1,000 
                                    samples and your application uses 
                                    two channels, the buffer size would 
                                    be 2,000 samples.
                                    see https://documentation.help/NI-DAQmx-Key-Concepts/bufferSize.html
                                    Requres that samples_per_channel > readback_freq // samples_per_channel

            terminal_config:        string, one of DEFAULT, DIFF, NRSE, PSEUDO_DIFF, RSE
                                    according to documentation: 
                                    https://nidaqmx-python.readthedocs.io/en/latest/constants.html#nidaqmx.constants.CalibrationTerminalConfig
                                    DEFAULT appears to be DIFF. Typically we want RSE (common ground)

            exp:                    string of experimenter names
            title:                  string for experiment title
            position:               string of position id

            settings:               kwargs to write to file header. Include items such as
                                        techron_gain
                                        pert_coil_ohms
                                        ref_coil_ohms
                                        note
                                    but takes general input. Saves as key: value
                                        in csv file header
        """

        # save inputs
        self.readback_freq = int(readback_freq)
        self.samples_per_channel = int(samples_per_channel)
        self.settings = settings
        self.terminal_config = getattr(
            ni.constants.TerminalConfiguration, terminal_config)
        self.exp = exp if exp is not None else ''
        self.title = title if title is not None else ''
        self.position = position if position is not None else ''

        self.t1 = settings['t1']
        self.t2 = settings['t2']
        self.t3 = settings['t3']
        self.chan = relay(0)

        # number of channels
        self.len_ai = len(self.ai)
        self.len_ao = len(self.ao)

        # samples per frame
        self._samples_per_frame = self.readback_freq // self._frames_per_buffer

        # make tasks
        self.taski = ni.Task()      # input (read)
        self.tasko = ni.Task()      # output (write)

        # setup read and write voltage channels
        for ch, _ in self.ai:
            self.taski.ai_channels.add_ai_voltage_chan(f"{self.device_name}{ch}",
                                                       **self.ai_args)
        for ch in self.ao:
            self.tasko.ao_channels.add_ao_voltage_chan(f"{self.device_name}{ch}",
                                                       **self.ao_args)

        # set terminal configuation
        self.taski.ai_channels.all.ai_term_cfg = self.terminal_config

        # setup clocks
        self.taski.timing.cfg_samp_clk_timing(rate=self.readback_freq,
                                              source=f'/{self.device_name}/ao/SampleClock',
                                              sample_mode=ni.constants.AcquisitionType.CONTINUOUS,
                                              samps_per_chan=self.samples_per_channel)
        self.tasko.timing.cfg_samp_clk_timing(rate=self.readback_freq,
                                              sample_mode=ni.constants.AcquisitionType.CONTINUOUS,
                                              samps_per_chan=self.samples_per_channel)

        # get streams
        self.stream_in = stream_readers.AnalogMultiChannelReader(
            self.taski.in_stream)
        self.stream_out = stream_writers.AnalogMultiChannelWriter(
            self.tasko.out_stream)

        # setup output buffer
        self.tasko.out_stream.output_buf_size = self.readback_freq

        # setup reading callback
        # read data when n samples are placed into the buffer
        self.taski.register_every_n_samples_acquired_into_buffer_event(self.samples_per_channel,
                                                                       self._read_task_callback)
        # get generator for output signal
        self.signal_generator = self._sine_generator(
            self._samples_per_frame, self.t1, self.t2, self.t3, self.chan)

        # setup output callback
        self.tasko.register_every_n_samples_transferred_from_buffer_event(self._samples_per_frame,
                                                                          self._write_task_callback)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _draw_in_progress(self, xdata, ydata):
        """
            Update a figure while the run is ongoing

            xdata: 1d np array of time stamps
            ydata: 2d np array of voltages to draw
        """

        # get obj
        ax = self.ax
        fig = ax.figure

        # set data
        for y, line in zip(ydata, ax.lines):
            line.set_data(xdata, y)

        # set limits
        n = int(len(ydata)/2)
        ax.set_ylim([np.min(ydata[:][n:]), np.max(ydata[:][n:])])

        # update canvas
        fig.canvas.draw()
        fig.canvas.flush_events()

    def _read_task_callback(self, task_handle, every_n_samples_event_type,
                            number_of_samples, callback_data):
        """
            Read data callback.

            Set up for a register_every_n_samples_acquired_into_buffer_event
            event.

            task_handle:                handle to the task on which the event occurred.
            every_n_samples_event_type: EveryNSamplesEventType.ACQUIRED_INTO_BUFFER value.
            number_of_samples parameter:the value you passed in the sample_interval parameter
                                        of the register_every_n_samples_acquired_into_buffer_event.
            callback_data:              apparently unused, but required
        """
        # set up buffer to read, needs to be the same size as data, for append to work
        buffer_in = np.empty((self.data.shape[0], self.samples_per_channel))

        # read samples
        self.stream_in.read_many_sample(buffer_in, number_of_samples,
                                        timeout=ni.constants.WAIT_INFINITELY)

        # appends buffered data to total variable data
        self.data = np.append(self.data, buffer_in, axis=1)

        # Absolutely needed for this callback to be well defined (see nidaqmx doc).
        return 0

    def _rebin(self, n):
        """
            Rebin self.df by factor n.
        """
        n = int(n)

        # easy end condition
        if n <= 1:
            return self.df

        # reset index
        df = self.df.reset_index()

        # rebin
        df = df.groupby(df.index // n).mean()

        # set time inedex back
        df.set_index('time (s)', inplace=True)

        return df

    def _sine_generator(self, array_length, t1, t2, t3, chan):
        """
            Construct a generator for sine waves to send to various outputs
            array_length:   number of samples in the output array such that the 
                            shape is (len_ao, array_length))
        """
        #self.chan = relay(0)
        # track phase in time domain
        # phase = 0
        # phase_step = array_length/self.readback_freq

        # time steps from t = 0
        t = np.arange(array_length, dtype=np.float64) / self.readback_freq
        
        print('t=',t)

        # time_start1 = time.time()
        frame_nr = 0
        refresh_rate_hz = self._frames_per_buffer
        
        while True:
            
            time = t + (frame_nr/refresh_rate_hz)
          
            yield np.array([self.amp[channel] * np.sin(2*np.pi*self.freq[channel]*(time))
                           for channel in range(self.len_ao)])
           
            
            frame_nr += 1
            
       

    def _write_task_callback(self, task_handle, every_n_samples_event_type,
                             number_of_samples, callback_data):
        """
            write data callback.

            Set up for a register_every_n_samples_transferred_from_buffer_event
            event.

            task_handle:                handle to the task on which the event occurred.
            every_n_samples_event_type: EveryNSamplesEventType.ACQUIRED_INTO_BUFFER value.
            number_of_samples parameter:the value you passed in the sample_interval parameter
                                        of the register_every_n_samples_acquired_into_buffer_event.
        """

        # generate signal
        signal = next(self.signal_generator)

        # save signal for debugging
        if self.save_signal:
            self.sine.append(signal)

        # write signal to device
        self.stream_out.write_many_sample(signal, timeout=1)

        # Absolutely needed for this callback to be well defined (see nidaqmx doc).
        return 0

    def close(self):
        """Close tasks"""
        self.taski.close()
        self.tasko.close()

    def draw_data(self, cols=None, **df_plot_kw):
        """
            Draw data in axis

            cols: list of column names to draw. If none, draw all

        """

        if cols is not None:
            self.df[cols].plot(**df_plot_kw)
        else:
            self.df.plot(**df_plot_kw)

        # plot elements
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage Readback (V)')
        plt.legend(fontsize='x-small')

    def draw_sine_generator(self, ax=None, **plot_kw):
        """
            Draw from the sine wave generator and draw it to the axes
        """

        # check that data is saved
        if not self.save_signal:
            raise RuntimeError("Must have save_signal = True")

        # draw in new fig
        if ax == None:
            plt.figure()
            ax = plt.gca()

        # get data from lists
        sine_data = []
        for i in range(len(self.sine[0])):
            sine_data.append(np.concatenate([s[i] for s in self.sine]))

        # time stamps
        x = np.arange(len(sine_data[0]))/self.readback_freq

        # draw
        for i, ch in enumerate(self.ao):
            ax.plot(x, sine_data[i], ls='--', label=ch.replace('/', ''),
                    **plot_kw)

        # plot elements
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage Readback')
        plt.legend(fontsize='x-small')

    def run(self, duration, freq=None, amp=None, draw_s=0, save_signal=False):
        """
            Take data, inputs are sine parameters

            run duration:   int, seconds
            freq:           float list, ao0 and ao1, if none, use from last run
            amp:            float list, ao0 and ao1, if none, use from last run
            draw_s:         if > 0, draw while in progress the last draw_s seconds
            save_signal:    if true, save signal output for later draw.
                            May crash the run if too long, very memory intensive 
                                and append gets slow at long list lengths
        """

        self.duration = duration
        self.save_signal = save_signal

        # sine wave parameters
        if freq is not None:
            self.freq = freq
        if amp is not None:
            self.amp = amp

        # reset data
        self.sine = []
        self.data = np.zeros((self.len_ai, self.samples_per_channel))

        # number of points to draw
        ndraw = draw_s*self.readback_freq if draw_s >= 0 else 0
        
        
        # initial fill of empty buffer (required else error)
        for _ in range(self._frames_per_buffer):
            
            self._write_task_callback(None, None, None, None)

        # start figure for drawing
        if draw_s:
            self.ax = plt.axes()
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Voltage (V)')
            self.ax.set_xlim(-draw_s, 0)
            for d, (_, ch) in zip(self.data, self.ai):
                self.ax.plot([0], [0], label=ch)
            self.ax.figure.legend(fontsize='xx-small')
            self.ax.figure.canvas.draw()
            plt.pause(0.1)
            plt.tight_layout()
            self.ax.figure.canvas.manager.window.attributes('-topmost', False)

        # start tasks (begin run)
        self.tasko.start()
        self.taski.start()

        time_start = time.time()
        dt = time_start - time.time()
        time_current = time_start
        progress_bar = tqdm(total=duration, leave=False,
                            desc=f'{self.freq[0]} Hz')
        try:

            while time_current-time_start < duration:

                # draw if needed
                if draw_s:
                    try:
                        ydata = self.data[:, self.samples_per_channel:]
                        n = int(min(len(ydata[0])-1, ndraw))

                        xdata = -1*np.arange(n)/self.readback_freq
                        ydata = ydata[:, -n:]
                        self._draw_in_progress(xdata[::-1], ydata)
                    except ValueError:
                        pass
                    
                    # open the relay here
                    
                    if time_current-time_start > duration - self.t3/2:
                        chan = relay()
                        chan.all_on()
                    
                # progress
                time_prev = time_current
                time_current = time.time()
                dt = time_current - time_prev
                progress_bar.update(dt)

        # if error, close task nicely
        except Exception as err:
            self.tasko.close()
            self.taski.close()
            raise err from None


        # time.sleep(self.tot_dur + 10)
        # stop task
        self.tasko.stop()
        self.taski.stop()

        # cleanup data: remove initial empty array, which we appended to
        self.data = self.data[:, self.samples_per_channel:]

        # draw
        if draw_s:
            try:
                ydata = self.data[:, self.samples_per_channel:]
                n = int(min(len(ydata[0])-1, ndraw))

                xdata = -1*np.arange(n)/self.readback_freq
                ydata = ydata[:, -n:]
                self._draw_in_progress(xdata[::-1], ydata)
            except ValueError:
                pass

        # reassign data to dataframe
        self.df = pd.DataFrame(
            {ch[1]: self.data[i] for i, ch in enumerate(self.ai) if 'null' not in ch[1]})
        self.df.index /= self.readback_freq
        self.df.index.name = 'time (s)'

    def to_csv(self, filename=None, save_dir='.', rebin=1):
        """
            Write to file in the same directory as this file
            if filename == None, generate default filename

            save_dir:   string, save directory relative to the location of this file OR 
                        list of strings to indicate subdirectories

            rebin:      write out average this many points to file to reduce file sizes
        """

        # check if noise run: no output amplitudes
        is_noise = self.amp[0] == 0

        # generate default filename
        if filename is None:
            filename = [f'{self.position}_Noise' if is_noise else f'{self.position}',
                        datetime.now().strftime('%y%m%dT%H%M%S'),
                        'p',
                        f'{self.freq[0]:g}Hz',
                        f'{self.amp[0]:g}V',
                        'r',
                        f'{self.duration:g}s'
                        '.csv']
            filename = '_'.join(filename)

        # ensure filename format
        filename = os.path.splitext(filename)[0]
        filename = filename + '.csv'

        # make path
        path = os.path.dirname(os.path.abspath(__file__))

        if type(save_dir) is str:
            path = os.path.join(path, save_dir)
        else:
            path = os.path.join(path, *save_dir)

        os.makedirs(path, exist_ok=True)

        # make full filename with path
        filename = os.path.join(path, os.path.basename(filename))

        # format physical settings
        if self.settings:
            settings = ['# Physical settings:']
            string_len = max([len(s) for s in self.settings.keys()]) + 2
            settings.extend(
                [f'#    {key:{string_len}}: {value}' for key, value in self.settings.items()])
        else:
            settings = []

        # header lines
        header = [f'# {self.title}',
                  '#',
                  f'# Fluxgate position: {self.position}',
                  '#',
                  '# Sine wave settings',
                  '#     Perturbation coil',
                  f'#         Freq:               {self.freq[0]} Hz',
                  f'#         NIDAQ amplitude:    {self.amp[0]} V',
                  '#     Reference coil',
                  '#',
                  f'# Run duration: {self.duration} s',
                  f'# Experimenters: {self.exp}',
                  *settings,
                  '#',
                  f'# File written by: {__file__}',
                  f'# {str(datetime.now())}',
                  '# \n']
        with open(filename, 'w') as fid:
            fid.write('\n'.join(header))

        # write data and rebin
        self._rebin(rebin).to_csv(filename, mode='a', index=True)
