"""filtering.py

Temporal filtering for Jagular. We assume that the original data is in (multiple) files
and that they are annoyingly large. So all the methods here work on buffered input,
using memory maps.

This work is based loosely on similar work by Kaushik Ghose. The original work can be found here:
https://github.com/kghose/neurapy/blob/master/neurapy/signal/continuous.py
"""

# AllDataArr = np.memmap(DatFileName,dtype=np.int16,shape=(n_samples,n_ch_dat),mode='r')
# b,a = signal.butter(3,100./(SAMPLE_RATE/2),'high') #filter at 100 Hz
# IntraArr = AllDataArr[:,IntraChannel].copy()
# IntraArr = signal.filtfilt(b,a,IntraArr)
# Thresh = IntraArr.max()*THRESH_FRAC

import numpy as np

from numpy import memmap
from scipy.signal import sosfiltfilt, iirdesign
from .utils import get_contiguous_segments

#Some useful presets
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


"""Complete IIR digital and analog filter design.
    Given passband and stopband frequencies and gains, construct an analog or
    digital IIR filter of minimum order for a given basic type.  Return the
    output in numerator, denominator ('ba'), pole-zero ('zpk') or second order
    sections ('sos') form.
    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.
        For digital filters, these are normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`wp` and `ws` are thus in
        half-cycles / sample.)  For example:
            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]
        For analog filters, `wp` and `ws` are angular frequencies (e.g. rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:
            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba'.
    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.
        """


def filtfilt_mmap(timestamps, finname, foutname, fs, fl=None, fh=None, gpass=None, gstop=None, dtype=None,
                  ftype='cheby2', buffer_len=16777216, overlap_len=None, max_len=-1):
    """Given sampling frequency, low and high pass frequencies design a chebyshev Type II filter, and filter our data with it."""

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
    except:
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

    overlap_len
    y = filtfilt_within_epochs_mmap(timestamps, finname, foutname, dtype, sos, buffer_len, overlap_len, max_len)
    return y

def filtfilt_within_epochs_mmap(timestamps, finname, foutname, dtype, sos, buffer_len=16777216, overlap_len=None, max_len=-1):
    """Use memmap and chunking to filter continuous data within contiguous segments.
    Inputs:
    finname -
    foutname    -
    dtype         - data format eg 'i'
    sos         - filter second order segments
    buffer_len  - how much data to process at a time
    overlap_len - how much data do we add to the end of each chunk to smooth out filter transients
    max_len     - how many samples to process. If set to -1, processes the whole file
    Outputs:
    y           - The memmapped array pointing to the written file
    Notes on algorithm:
    1. The arrays are memmapped, so we let pylab (numpy) take care of handling large arrays
    2. The filtering is done in chunks:
    Chunking details:
                |<------- b1 ------->||<------- b2 ------->|
    -----[------*--------------{-----*------]--------------*------}----------
            |<-------------- c1 -------------->|
                                |<-------------- c2 -------------->|
    From the array of data we cut out contiguous buffers (b1,b2,...) and to each buffer we add some extra overlap to
    make chunks (c1,c2). The overlap helps to remove the transients from the filtering which would otherwise appear at
    each buffer boundary.
    """
    x = memmap(finname, dtype=dtype, mode='r')
    if (max_len == -1) or (max_len is None):
        max_len = x.size
    try:
        y = memmap(foutname, dtype=dtype, mode='w+', shape=max_len)
    except OSError:
        raise ValueError('Not sure why this ODError is raised, actually? File already exists?')

    epochs = get_contiguous_segments(timestamps)
    ccr = np.cumsum(epochs[:,1] - epochs[:,0]).astype(np.int64)
    filter_epochs = np.vstack((np.insert(ccr[:-1],0,0),ccr)).T

    for (start, stop) in filter_epochs:
        for buff_st_idx in range(start, stop, buffer_len):
            chk_st_idx = max(start, buff_st_idx - overlap_len)
            buff_nd_idx = min(stop, buff_st_idx + buffer_len)
            chk_nd_idx = min(stop, buff_nd_idx + overlap_len)
            rel_st_idx = buff_st_idx - chk_st_idx
            rel_nd_idx = buff_nd_idx - chk_st_idx
#             print('filtering {}--{}'.format(chk_st_idx, chk_nd_idx))
            this_y_chk = sosfiltfilt(sos, x[chk_st_idx:chk_nd_idx])
#             print('saving {}--{}'.format(buff_st_idx, buff_nd_idx))
            y[buff_st_idx:buff_nd_idx] = this_y_chk[rel_st_idx:rel_nd_idx]

    return y