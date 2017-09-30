"""Split raw channels into seperate files."""

import struct
import glob

import jagular as jag #TODO: don't import externally

from contextlib import ExitStack

# file_list = ['../sample_data/sample_data_1.rec',
#              '../sample_data/sample_data_3.rec',
#              '../sample_data/sample_data_4.rec',
#              '../sample_data/sample_data_5.rec',
#              '../sample_data/sample_data_2.rec'
#             ]

file_list = glob.glob('*.rec')

jfm = jag.io.JagularFileMap(file_list)

block_size = 65536

n_chan_zfill = len(str(jfm._reader.n_spike_channels)) # xxx for 100s of channels, xx for 10s of channels, xxxx for 1000s etc.
ch_out_prefix = ''
ch_out_files = [ch_out_prefix + 'ch.' + str(n).zfill(n_chan_zfill) + '.raw' for n in range(jfm._reader.n_spike_channels)]

#TODO: make filenames more configurable
#TODO: warn if files already exist, or if we cannot create them (this latter one should be handled automatically)

with ExitStack() as stack:
    ts_file = stack.enter_context(open('timestamps.raw', 'wb+'))
    ch_files = [stack.enter_context(open(fname, 'wb+')) for fname in ch_out_files]

    for ii, (ts, all_ch_data) in enumerate(jfm.read_stitched_files(block_size=block_size)):
        num_packets = len(ts)

        my_ts_struct = struct.Struct('<%dI' % num_packets)
        my_ch_struct = struct.Struct('<%dh' % num_packets)
        ts_packed = my_ts_struct.pack(*ts)
        for ch in range(jfm._reader.n_spike_channels):
            ch_packed = my_ch_struct.pack(*all_ch_data[ch,:])
            # write current channel data of current block to file:
            ch_files[ch].write(ch_packed)

        # write timestamps of current block to file:
        ts_file.write(ts_packed)

