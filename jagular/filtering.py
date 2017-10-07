"""filtering.py

Temporal filtering for Jagular. We assume that the original data is in (multiple) files
and that they are annoyingly large. So all the methods here work on buffered input,
using memory maps.

This work is based on similar work by Kaushik Ghose. The original work can be found here:
https://github.com/kghose/neurapy/blob/master/neurapy/signal/continuous.py
"""

# AllDataArr = np.memmap(DatFileName,dtype=np.int16,shape=(n_samples,n_ch_dat),mode='r')
# b,a = signal.butter(3,100./(SAMPLE_RATE/2),'high') #filter at 100 Hz
# IntraArr = AllDataArr[:,IntraChannel].copy()
# IntraArr = signal.filtfilt(b,a,IntraArr)
# Thresh = IntraArr.max()*THRESH_FRAC

from numpy import memmap
from scipy.signal import filtfilt, iirdesign

#Some useful presets
spikegadgets_lfp = {
    'dtype': 'np.int16',
    'ts_dtype': 'np.uint32',
    'fs' : 30000,  # sampling rate [Hz]
    'fl' : None,      # low cut for spike filtering
    'fh' : None,    # high cut for spike filtering
    'gpass' : 0.1, # maximum loss in the passband (dB)
    'gstop' : 15,  # minimum attenuation in the stopband (dB)
    'buffer_len' : 1048576, # number of samples to process at a time (1048576 = 1024^2)
    'overlap_len': 65536,   # number of samples to overlap, in each direction (65536 = 256^2)
    'max_len': None
}

spikegadgets_spike = {
    'dtype': 'np.int16',
    'ts_dtype': 'np.uint32',
    'fs' : 30000,  # sampling rate [Hz]
    'fl' : 500,    # low cut for spike filtering
    'fh' : 8000,   # high cut for spike filtering
    'gpass' : 0.1, # maximum loss in the passband (dB)
    'gstop' : 15,  # minimum attenuation in the stopband (dB)
    'buffer_len' : 1048576, # number of samples to process at a time (1048576 = 1024^2)
    'overlap_len': 65536,   # number of samples to overlap, in each direction (65536 = 256^2)
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

def butterfilt(finname, foutname, dtype, fs, fl=5.0, fh=100.0, gpass=1.0, gstop=30.0, ftype='butter', buffer_len=100000, overlap_len=100, max_len=-1):
    """Given sampling frequency, low and high pass frequencies design a butterworth filter, and filter our data with it."""
    fso2 = fs/2.0
    wp = [fl/fso2, fh/fso2]
    ws = [0.8*fl/fso2,1.4*fh/fso2]
    b, a = iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype=ftype, output='ba')
    y = filtfiltlong(finname, foutname, dtype, b, a, buffer_len, overlap_len, max_len)
    return y, b, a

def filtfiltlong(finname, foutname, dtype, b, a, buffer_len=100000, overlap_len=100, max_len=-1):
    """Use memmap and chunking to filter continuous data.
    Inputs:
    finname -
    foutname    -
    dtype       - data format eg 'np.int16'
    b,a         - filter coefficients
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
    if max_len == -1:
        max_len = x.size
    y = memmap(foutname, dtype=dtype, mode='w+', shape=max_len)
    # for each epoch
        # determine start, stop
    for buff_st_idx in range(0, max_len, buffer_len):
        chk_st_idx = max(0, buff_st_idx - overlap_len)
        buff_nd_idx = min(max_len, buff_st_idx + buffer_len)
        chk_nd_idx = min(x.size, buff_nd_idx + overlap_len)
        rel_st_idx = buff_st_idx - chk_st_idx
        rel_nd_idx = buff_nd_idx - chk_st_idx
        this_y_chk = filtfilt(b, a, x[chk_st_idx:chk_nd_idx], method="gust")
        y[buff_st_idx:buff_nd_idx] = this_y_chk[rel_st_idx:rel_nd_idx]

    return y