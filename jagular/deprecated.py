# deprecated.py

def decimate_within_epochs(tsfinname, finname, foutname, dtype, sos,
                           buffer_len=16777216, max_len=None):
    """Docstring goes here.

    This is TEMPORARY! Not how Jagular should be used! But we need resultzzz.

    So, this function takes in a memmapped raw channel (say 15 GB) and an
    IN-CORE timestamps file, and spits out a dumbsampled version, taking every
    10th sample in addition to first and last samples within each contiguous
    block (epoch).
    """

    x = memmap(finname, dtype=dtype, mode='r') # input channel data, possibly out-of-core
    ts_dtype = np.uint32
    timestamps = np.fromfile(tsfinname, dtype=ts_dtype)

    timestamp_epochs = get_contiguous_segments(timestamps) # epochs in timestamps
    ccr = np.cumsum(timestamp_epochs[:,1] - timestamp_epochs[:,0]).astype(np.int64)
    index_epochs = np.vstack((np.insert(ccr[:-1],0,0),ccr)).T # epochs in indices

    for (start, stop) in index_epochs:
        for buff_st_idx in range(start, stop, buffer_len):
            chk_st_idx = max(start, buff_st_idx)
            chk_nd_idx = min(stop, buff_st_idx + buffer_len)
            rel_st_idx = buff_st_idx - chk_st_idx
            rel_nd_idx = buff_nd_idx - chk_st_idx
            print('dumbsampling {}--{}'.format(chk_st_idx, chk_nd_idx))
            >>> x[chk_st_idx:chk_nd_idx])
            print('saving {}--{}'.format(buff_st_idx, buff_nd_idx))
            y[buff_st_idx:buff_nd_idx] = this_y_chk[rel_st_idx:rel_nd_idx]

    return y

with ExitStack() as stack:
        ts_file = stack.enter_context(open(ts_out, 'wb+'))
        ch_files = [stack.enter_context(open(fname, 'wb+')) for fname in ch_out_files]

        for ii, (ts, all_ch_data) in enumerate(jfm.read_stitched_files(block_size=block_size)):

my_ch_struct = Struct('<%dh' % num_packets)
my_ts_struct = Struct('<%dI' % num_packets) # ts_dtype should affect this!!!!

ts_packed = my_ts_struct.pack(*ts)
ts_file.write(ts_packed)
