"""Split raw channels into seperate files."""

import glob
import matplotlib.pyplot as plt
import numpy as np

import jagular as jag #TODO: don't import externally

import nelpy as nel
import nelpy.plotting as npl

# file_list = ['../sample_data/sample_data_1.rec',
#              '../sample_data/sample_data_3.rec',
#              '../sample_data/sample_data_4.rec',
#              '../sample_data/sample_data_5.rec',
#              '../sample_data/sample_data_2.rec'
#             ]

# file_list = glob.glob('sample_data/*.rec')   # TODO: this fails the timstamp checks... is this correct?
file_list = glob.glob('sample_data/gap_data.rec')
jfm = jag.io.JagularFileMap(file_list)

# extract only a subset of channels, interpolating over gaps of 200 samples or less:
jag.utils.extract_channels(jfm=jfm,
                           max_gap_size=200,
                           ch_out_prefix='channels/subset_',
                           subset=[9,3,0])

# extract all channels, with no interpolation:
jag.utils.extract_channels(jfm=jfm,
                           ts_out='timestamps_new.raw',
                           ch_out_prefix='channels/',
                           verbose=True)

ts1 = np.fromfile('timestamps.raw', dtype=np.uint32)
x1 = np.fromfile('channels/subset_ch.00.raw', dtype=np.int16)

ts2 = np.fromfile('timestamps_new.raw', dtype=np.uint32)
x2 = np.fromfile('channels/ch.00.raw', dtype=np.int16)

plt.plot(ts1,x1, lw=5, c='0.3')
plt.plot(ts2,x2, lw=1, c='w')

ts = np.fromfile('timestamps.raw', dtype=np.uint32)
y = np.fromfile('channels/subset_ch.00.raw', dtype=np.int16)
sig = nel.AnalogSignalArray(ydata=y, timestamps=ts/30000, fs=30000)
print(sig)
npl.plot(sig)

ts = np.fromfile('timestamps_new.raw', dtype=np.uint32)
y = np.fromfile('channels/ch.00.raw', dtype=np.int16)
sig = nel.AnalogSignalArray(ydata=y, timestamps=ts/30000, fs=30000)
print(sig)
npl.plot(sig)

# now filter some data!
timestamps = np.fromfile('timestamps.raw', dtype=np.uint32)
y = jag.filtfilt_mmap(timestamps=timestamps,
             finname='channels/subset_ch.00.raw',
             buffer_len=167600,
             foutname='filtered2.raw',
             fs=30000,
             fl=7,
             fh=9)

from .filtering import spikegadgets_spike_filter_params

y = jag.filtfilt_mmap(timestamps=timestamps,
                      finname='channels/subset_ch.00.raw',
                      foutname='filtered2.raw',
                      **spikegadgets_spike_filter_params)
