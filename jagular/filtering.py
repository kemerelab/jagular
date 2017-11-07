"""filtering.py

Temporal filtering for Jagular. We assume that the original data is in (multiple) files
and that they are annoyingly large. So all the methods here work on buffered input,
using memory maps.

This work is based loosely on similar work by Kaushik Ghose. The original work can be found here:
https://github.com/kghose/neurapy/blob/master/neurapy/signal/continuous.py
"""

import numpy as np

from numpy import memmap
from scipy.signal import sosfiltfilt, iirdesign
from .utils import _get_contiguous_segments_fast as get_contiguous_segments

def filtfilt_mmap(timestamps, finname, foutname, fs, fl=None, fh=None,
                  gpass=None, gstop=None, dtype=None, ftype='cheby2',
                  buffer_len=4194304, overlap_len=None, max_len=None,
                  **kwargs):
    """Zero-phase forward backward out-of-core Chebyshev type II filter.

    Parameters
    ----------
    timestamps : array-like
        DESCRIPTION GOES HERE
    finname : str
        DESCRIPTION GOES HERE
    foutname : str
        DESCRIPTION GOES HERE
    fs : float
        The sampling frequency (Hz).
    fl : float, optional
        Low cut-off frequency (in Hz), 0 or None to ignore. Default is None.
    fh : float, optional
        High cut-off frequency (in Hz), 0 or None to ignore. Default is None.
    gpass : float, optional
        The maximum loss in the passband (dB). Default is 0.1 dB.
    gstop : float, optional
        The minimum attenuation in the stopband (dB). Default is 30 dB.
    dtype : datatype for channel data, optional
        DESCRIPTION GOES HERE. Default np.int16
    ftype : str, optional
        The type of IIR filter to design:
            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2' (Default)
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'
    buffer_len : int, optional
        How much data to process at a time. Default is 2**22 = 4194304 samples.
    overlap_len : int, optional
        How much data do we add to the end of each chunk to smooth out filter
        transients
    max_len : int, optional
        When max_len == -1 or max_len == None, then argument is effectively
        ignored. If max_len is a positive integer, thenmax_len specifies how
        many samples to process.

    Returns
    -------
    y : numpy.memmmap
        Numpy memmap reference to filtered object.
    """

    if overlap_len is None:
        overlap_len = int(fs*2)

    if dtype is None:
        dtype=np.int16

    if gpass is None:
        gpass = 0.1 # max loss in passband, dB

    if gstop is None:
        gstop = 30 # min attenuation in stopband (dB)

    fso2 = fs/2.0

    try:
        if np.isinf(fh):
            fh = None
    except AttributeError:
        pass
    if fl == 0:
        fl = None

    if (fl is None) and (fh is None):
        print('wut? nothing to filter, man!')
        raise ValueError('nonsensical all-pass filter requested...')
    elif fl is None: # lowpass
        wp = fh/fso2
        ws = 1.4*fh/fso2
    elif fh is None: # highpass
        wp = fl/fso2
        ws = 0.8*fl/fso2
    else: # bandpass
        wp = [fl/fso2, fh/fso2]
        ws = [0.8*fl/fso2,1.4*fh/fso2]

    sos = iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype=ftype, output='sos')

    y = filtfilt_within_epochs_mmap(timestamps=timestamps,
                                    finname=finname,
                                    foutname=foutname,
                                    dtype=dtype,
                                    sos=sos,
                                    buffer_len=buffer_len,
                                    overlap_len=overlap_len,
                                    max_len=max_len,
                                    **kwargs)
    return y

def filtfilt_within_epochs_mmap(timestamps, finname, foutname, dtype, sos,
                                buffer_len=4194304, overlap_len=None,
                                max_len=None, filter_epochs=None,**kwargs):
    """Zero-phase forward backward out-of-core filtering within epochs.

    Use memmap and chunking to filter continuous data within contiguous segments

    Parameters
    ----------
    timestamps : array-like
        DESCRIPTION GOES HERE
    finname : str
        DESCRIPTION GOES HERE
    foutname : str
        DESCRIPTION GOES HERE
    dtype : datatype for channel data
        DESCRIPTION GOES HERE
    sos : ndarray
        Second-order sections representation of the IIR filter.
    buffer_len : int, optional
        How much data to process at a time. Default is 2**22 = 4194304 samples.
    overlap_len : int, optional
        How much data do we add to the end of each chunk to smooth out filter
        transients
    max_len : int, optional
        When max_len == -1 or max_len == None, then argument is effectively
        ignored. If max_len is a positive integer, thenmax_len specifies how
        many samples to process.

    Returns
    -------
    y : numpy.memmmap
        Numpy memmap reference to filtered object.

    Notes on algorithm
    ------------------
    1. The arrays are memmapped, so we let numpy take care of handling large
       arrays
    2. The filtering is done in chunks:
    Chunking details:
                |<------- b1 ------->||<------- b2 ------->|
    -----[------*--------------{-----*------]--------------*------}----------
            |<-------------- c1 -------------->|
                                |<-------------- c2 -------------->|
    From the array of data we cut out contiguous buffers (b1,b2,...) and to each
    buffer we add some extra overlap to make chunks (c1,c2). The overlap helps
    to remove the transients from the filtering which would otherwise appear at
    each buffer boundary.
    """

    x = memmap(finname, dtype=dtype, mode='r')
    if (max_len == -1) or (max_len is None):
        max_len = x.size
    try:
        y = memmap(foutname, dtype=dtype, mode='w+', shape=max_len)
    except OSError:
        raise ValueError('Not sure why this ODError is raised, actually? File already exists?')

    # TODO: maybe defaults of assume_sorted=True and step=1 are too lenient? rethink the API slightly...
    assume_sorted = kwargs.get('assume_sorted', True)
    step = kwargs.get('step', 1)

    if filter_epochs is None:
        filter_epochs = get_contiguous_segments(data=timestamps,
                                                assume_sorted=assume_sorted,
                                                step=step,
                                                index=True)

    for (start, stop) in filter_epochs:
        for buff_st_idx in range(start, stop, buffer_len):
            chk_st_idx = int(max(start, buff_st_idx - overlap_len))
            buff_nd_idx = int(min(stop, buff_st_idx + buffer_len))
            chk_nd_idx = int(min(stop, buff_nd_idx + overlap_len))
            rel_st_idx = int(buff_st_idx - chk_st_idx)
            rel_nd_idx = int(buff_nd_idx - chk_st_idx)
#             print('filtering {}--{}'.format(chk_st_idx, chk_nd_idx))
            this_y_chk = sosfiltfilt(sos, x[chk_st_idx:chk_nd_idx])
#             print('saving {}--{}'.format(buff_st_idx, buff_nd_idx))
            y[buff_st_idx:buff_nd_idx] = this_y_chk[rel_st_idx:rel_nd_idx]

    return y

# Some useful presets
spikegadgets_lfp_filter_params = {
    'dtype': np.int16,
    # 'ts_dtype': 'np.uint32',
    'fs' : 30000,  # sampling rate [Hz]
    'fl' : None,      # low cut for spike filtering
    'fh' : None,    # high cut for spike filtering
    'gpass' : 0.1, # maximum loss in the passband (dB)
    'gstop' : 30,  # minimum attenuation in the stopband (dB)
    'buffer_len' : 16777216, # number of samples to process at a time (16777216 = 2**24)
    'overlap_len': 65536,   # number of samples to overlap, in each direction (65536 = 2**16)
    'max_len': None
}

spikegadgets_spike_filter_params = {
    'dtype': np.int16,
    # 'ts_dtype': 'np.uint32',
    'fs' : 30000,  # sampling rate [Hz]
    'fl' : 600,    # low cut for spike filtering
    'fh' : 6000,   # high cut for spike filtering
    'gpass' : 0.1, # maximum loss in the passband (dB)
    'gstop' : 30,  # minimum attenuation in the stopband (dB)
    'buffer_len' : 16777216, # number of samples to process at a time (16777216 = 2**24)
    'overlap_len': 65536,   # number of samples to overlap, in each direction (65536 = 2**16)
    'max_len': None
}
"""Use these presets as follows
from jagular import filtering as jfilt
y, b, a = jfilt.butterfilt(*files, ofile='test.raw', **jfilt.spikegadgets_spike)"""
