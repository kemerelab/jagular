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

#######################################################

max_gap_size = 200

ch_out_prefix = 'channels/gap_'
n_chan_zfill = len(str(jfm._reader.n_spike_channels))
ch_out_files = [ch_out_prefix + 'ch.' + str(n).zfill(n_chan_zfill) + '.raw' for n in range(jfm._reader.n_spike_channels)]

from contextlib import ExitStack
from scipy.interpolate import interp1d

prev_channel_data = None # used for across-block interpolation
prev_ts_data = None      # used for across-block interpolation
# assumption: block_size >> interp_size (we can check for this with an assert)

with ExitStack() as stack:
    ts_file = stack.enter_context(open('timestamps.raw', 'wb+'))
    ch_files = [stack.enter_context(open(fname, 'wb+')) for fname in ch_out_files]

    for ii, (ts, all_ch_data) in enumerate(jfm.read_stitched_files(block_size=800000)):
        # inspect timestamps, and interpolate if necessary...
        # TODO: add full sanitization here...

        if max_gap_size > 0:

            if prev_ts_data is not None:
                inter_block_gap = ts[0] - prev_ts_data
                if (inter_block_gap <= max_gap_size) & (inter_block_gap > 1):
                    print('we need to interpolate across blocks! (block {} to {}, sample {} to {})'.format(ii-1, ii, prev_ts_data, ts[0]))
                    pre_ts = np.arange(prev_ts_data, ts[0])
                    f = interp1d([prev_ts_data, ts[0]], np.vstack([prev_channel_data, all_ch_data[:,0]]).T, assume_sorted=True)
                    pre_ch = f(pre_ts) # in floats, not np.int16!
                    pre_ch = pre_ch.astype(np.int16)
                    print(pre_ch.shape)
                    print(all_ch_data.shape)
                    all_ch_data = np.hstack([pre_ch, all_ch_data])
                    print(all_ch_data.shape)
                    ts = np.hstack([pre_ts, ts])
                    print(ts)

            prev_ts_data = ts[-1]
            prev_channel_data = all_ch_data[:,-1]

            # now interpolate all interior qualifying regions of the block:
            # get gaps
            cs = jag.utils.get_contiguous_segments(ts).astype(np.int32)
            gap_lengths = cs[1:,0] - cs[:-1,1]

            if np.any(gap_lengths <= max_gap_size):
                # only do this if there are some gaps satisfying the criteria
                tt = np.argwhere(gap_lengths<=max_gap_size)
                vv = np.argwhere(gap_lengths>max_gap_size)
                ccl = (np.cumsum(cs[:,1] - cs[:,0]) - 1).astype(np.int32)
                ccr = np.cumsum(cs[:,1] - cs[:,0]).astype(np.int32)
                orig_data_locs = np.vstack((np.insert(ccr[:-1],0,0),ccr)).T # want this as seperate function, too!
                split_data_ts = []
                split_data = []
                for kk, (start, stop) in enumerate(orig_data_locs):
                    split_data_ts.append(cs[kk,0])
                    split_data.append(all_ch_data[:,start:stop])
                stops = cs[:,1]
                starts = cs[1:,0]

                interpl_ts = np.atleast_1d(stops[tt].squeeze())
                interpl_ch = np.atleast_2d(all_ch_data[:,ccl[tt]])
                interpl_ch = interpl_ch.squeeze(axis=2)

                interpr_ts = np.atleast_1d(starts[tt].squeeze())
                interpr_ch = np.atleast_2d(all_ch_data[:,ccr[tt]])
                interpr_ch = interpr_ch.squeeze(axis=2)

                new_cs = np.hstack((np.vstack((cs[0,0], starts[vv])), np.vstack((stops[vv], cs[-1,1]))))

                # generate new timestamps:
                ts_list = [list((nn for nn in range(start, stop))) for (start, stop) in new_cs]
                ts_new = [item for sublist in ts_list for item in sublist]

                for kk, (itsl, itsr) in enumerate(zip(interpl_ts, interpr_ts)):
                    # build interp object
                    f = interp1d([itsl-1, itsr], np.vstack([interpl_ch[:,kk], interpr_ch[:,kk]]).T, assume_sorted=True)
                    interp_ts = np.arange(itsl, itsr)
                    interp_ch = f(interp_ts) # in floats, not np.int16!
                    interp_ch = interp_ch.astype(np.int16)
                    split_data_ts.append(itsl)
                    split_data.append(interp_ch)

                # now reassemble split channeldata chunks in order:
                chunk_order = np.argsort(split_data_ts)
                ch_data_new = np.hstack([split_data[hh] for hh in chunk_order])

                all_ch_data = ch_data_new
                ts = ts_new

        # re-estimate number of packets
        num_packets = len(ts)
        my_ch_struct = struct.Struct('<%dh' % num_packets)
        my_ts_struct = struct.Struct('<%dI' % num_packets)
        ts_packed = my_ts_struct.pack(*ts)

        for ch in range(jfm._reader.n_spike_channels):
            ch_packed = my_ch_struct.pack(*all_ch_data[ch,:])
            # write current channel data of current block to file:
            ch_files[ch].write(ch_packed)

        # write timestamps of current block to file:
        ts_file.write(ts_packed)