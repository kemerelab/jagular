"""See https://github.com/mcleonard/spikesort/blob/master/process.py

also, https://github.com/mcleonard/spikesort for parallel, and some plots"""


"""
.. module:: process
    :synopsis:  This module is intended for processing raw electrical signals.  For
        example, to detect and extract spikes or other threshold crossings from
        an electrophysiology recording.

.. moduleauthor:: Mat Leonard <leonard.mat@gmail.com>
"""

import numpy as np

class ProcessingError(Exception):
    pass

def map(func, data, processes=4):
    """ This maps the data to func in parallel using multiple processes.
        This works fine in the IPython terminal, but not in IPython notebook.

        **Arguments**:
            *func*:
             Function to map the data with.

            *data*: Data sent to func.

        **Returns**:
            Returns whatever func returns.
    """

    from multiprocessing import Pool
    pool = Pool(processes=processes)
    output = pool.map(func, data)
    pool.close()
    pool.join()

    return output

def tetrode_chans(tetrode_num):
    """ Return the channel numbers for the requested tetrode.

    .. warning::
        These channels are only valid for the H04 and H05 adapters used in our
        lab.

    **Arguments**:
        *tetrode_num*:
         Tetrode number.

    **Returns**:
        A list of channels belonging to the chosen tetrode.

    """

    tetrodes = {1:[16,18,17,20], 2:[19,22,21,24], 3:[23,26,25,28],
                4:[27,30,29,32]}

    return tetrodes[tetrode_num]

def load_ns5(filename, channels=None):
    """ Load data from an ns5 file.

        This returns a generator, so you only get one channel at a time, but
        you don't have to load in all the channels at once, saving memory
        since it is a LOT of data.  This only works with ns5 files currently.

        **Arguments**:
            *filename*:
             A path to the data file.

        **Keywords**:
            *channels*:
             The channels to load.

        **Returns**:
            A generator of numpy arrays containing the raw voltage signal from
            the given channels.
    """

    import ns5

    loader = ns5.Loader(filename)
    loader.load_file()
    bit_to_V = 4096.0 / 2.**15 # uV/bit

    for chan in channels:
        yield loader.get_channel_as_array(chan)*bit_to_V

def common_ref(data, n=None):
    """ Calculates the common average reference from the data.

        Calculate the common average reference from the data to subtract from
        each channel of the raw data.  Doing this removes noise and artifacts
        from the raw data so that spike detection performance is improved. If
        the length of data can't be found with len(), set n to the length
        of data.

        **Arguments**:
            *data*:
             An iterator containing equal length arrays.

        **Keywords**:
            *n*:
             The number of arrays in data.

        **Returns**:
            A numpy array of the average of the arrays in data.
    """

    try:
        n = len(data)
    except TypeError:
        n = n
    return np.sum(data)/float(n)

def save_spikes(filename, spikes):
    """ Saves spikes record array to file. """

    with open(filename, 'w') as f:
        spikes.tofile(f)
    print('Saved to {}'.format(filename))

def load_spikes(filename, ncols=120):
    """ Loads recarray saved with save_spikes.  The keyword ncols should be
        set to the length of the spike waveform.
    """

    with open(filename, 'r') as f:
        loaded = np.fromfile(f)

    spikes = loaded.reshape(len(loaded)/(ncols+1), ncols+1)
    records = [('spikes', 'f8', ncols), ('times', 'f8', 1)]
    recarray = np.zeros(len(spikes), dtype = records)
    recarray['spikes'] = spikes[:,:120]
    recarray['times'] = spikes[:,-1]

    return recarray

def detect_spikes(data, threshold=4, patch_size=30, offset=0):
    """ Detect spikes in data.

    **Arguments**:
        *data*:
         The data to extract spikes from, should be a numpy array.

    **Keywords**:
        *threshold*:
         The threshold for spike detection, approximately equal the number of
         standard deviations of the noise.

        *patch_size*:
         The number of samples for an extracted spike patch.

    **Returns**:
        A numpy recarray with the following fields:

        *spikes*:
         An N x patch_size array of spike waveform patches, where N is the
         number of spikes detected.

        *times*:
         An array of time samples for the peak of each detected spike.

    """

    import time
    start = time.time()

    threshold = get_threshold(data, multiplier=threshold)
    peaks = crossings(data, threshold, polarity='neg')
    peaks = censor(peaks, 30)

    spikes, times = extract(data, peaks, patch_size=patch_size, offset=offset)

    records = [('spikes', 'f8', patch_size), ('times', 'f8', 1)]
    detected = np.zeros(len(times), dtype=records)
    detected['spikes'] = spikes
    detected['times'] = times

    elapsed = time.time() - start
    print("Detected {} spikes in {} seconds".format(len(times), elapsed))

    return detected

def form_tetrode(data, times, patch_size=30, offset=0, samp_rate=30000):
    """ Build tetrode waveforms from voltage data and detected spike times.

    **Arguments**:
        *data*:
         The voltage signals of each channel in a tetrode, used for
         extracting spikes.

        *times*:
         A numpy array containing the sample value for each spike.

    **Keywords**:
        *patch_size*:
         The number of samples to extract centered on peak + offset.

        *offset*:
         The number of samples to offset the extracted patch from peak.

        *samp_rate*:
         The sampling rate of the recorded data.

    **Returns**:
        A numpy recarray with fields:

        *spikes*:
         Arrays that are 4*patch_size long, containing waveforms
         from each data channel extracted at each time stamp from times.

        *times*:
         The same array as the times argument.

    """

    extracted = [ extract(chan, times,
                          patch_size=patch_size,
                          offset=offset)
                  for chan in data]
    waveforms = np.concatenate([ wv for wv, time in extracted ], axis=1)

    # Remove any spikes that are too large

    good_spikes = np.where((waveforms<300).all(axis=1) *
                           (waveforms>-300).all(axis=1))[0]

    records = [('spikes', 'f8', patch_size*4), ('times', 'f8', 1)]
    tetrodes = np.zeros(len(good_spikes), dtype=records)
    tetrodes['spikes'] = waveforms[good_spikes]
    tetrodes['times'] = times[good_spikes]/float(samp_rate)

    return tetrodes

def get_threshold(data, multiplier=4):
    """ Calculate the spike crossing threshold from the data.

    Uses the median of the given data to calculate the standard deviation of
    the noise.  This method is less sensitive to spikes in the data.

    **Arguments**:
        *data*:
         The data numpy array.

    **Keywords**:
        *multiplier*:
         The threshold multiplier, approximately the number of significant
         deviations of the noise.

    **Returns**:
        A float value for the threshold.

    """
    return multiplier*np.median(np.abs(data)/0.6745)

def filter(data, low=300, high=6000, rate=30000):
    """ Filter the data with a 3-pole Butterworth bandpass filter.

        This is used to remove LFP from the signal.  Also reduces noise due
        to the decreased bandwidth.  You will typically filter the raw data,
        then extract spikes.

        **Arguments**:
            *data*: The data you want filtered.

        **Keywords**:
            *low*:
             Low frequency rolloff.

            *high*:
             High frequency rolloff.

            *rate*:
             The sample rate.

        **Returns**:
            A numpy array the same shape as data, but filtered.

    """
    import scipy.signal as sig

    if high > rate/2.:
        high = rate/2.-1
        print("High rolloff frequency can't be greater than the Nyquist \
               frequency.  Setting high to {}").format(high)

    filter_lo = low #Hz
    filter_hi = high #Hz
    samp_rate = float(rate)

    #Take the frequency, and divide by the Nyquist freq
    norm_lo = filter_lo/(samp_rate/2)
    norm_hi = filter_hi/(samp_rate/2)

    # Generate a 3-pole Butterworth filter
    b, a = sig.butter(3, [norm_lo,norm_hi], btype="bandpass");
    return sig.filtfilt(b, a, data)

def censor(data, width=30):
    """ Censor values after leading edges in time.

    This is used to insert a censored period in threshold crossings.
    For instance, when you find a crossing in the signal, you don't
    want the next 0.5-1 ms, you just want the first crossing.

    **Arguments**:
        *data*:
         A numpy array, the data you want censored.

    **Keyword**:
        *width*:
         The number of samples censored after a leading edge.

    **Returns**:
        A numpy array of leading edge timestamps.

    **Example**:

    >>> times = [110, 111, 112, 120, 270, 271, 280]
    >>> censored = censor(times)
    >>> print(censored)
    array([110, 270])

    """
    try:
        edges = [data[0]]
    except IndexError:
        raise ValueError("data is empty")

    for sample in data:
        if sample > edges[-1] + width:
            edges.append(sample)
    return np.array(edges)

def crossings(data, threshold, polarity='pos'):
    """ Find threshold crossings in data.

    **Arguments**:
        *data*:
         A numpy array of the data.

        *threshold*:
         The voltage threshold, always positive.

    **Keywords**:
        *polarity* ('pos', 'neg', 'both'):

            * 'pos': detects crossings for +threshold
            * 'neg': detects crossings for -threshold
            * 'both': both + and - threshold

    **Returns**:
        An array of sample timestamps for each threshold crossing.

    This gives all samples that cross the threshold.  If you only want the
    first crossings, pass the results to :func:`censor`.

    """

    if ~isinstance(data, list):
        data = [data]
    peaks = []
    for chan in data:
        if polarity == 'neg' or polarity == 'both':
            below = np.where(chan<-threshold)[0]
            peaks.append(below[np.where(np.diff(below)==1)])
        elif polarity == 'pos' or polarity == 'both':
            above = np.where(chan>threshold)[0]
            peaks.append(above[np.where(np.diff(above)==1)])

    return np.concatenate(peaks)

def extract(data, peaks, patch_size=30, offset=0, polarity='neg'):
    """ Extract peaks from data based on sample values in peaks.

        **Arguments**:
            *data*:
             The data you want to extract patches from.

            *peaks*:
             The sample timestamps where the patches are taken from.

        **Keywords**:
            *patch_size*:
             The number of samples to extract centered on peak + offset.

            *offset*:
             The number of samples to offset the extracted patch from peak.

            *polarity*:
             ('pos' or 'neg') Set to 'pos' if your spikes have positive polarity

        **Returns**:
            *spikes*:
             A len(peaks) x patch_size array of extracted spikes.

            *peaks*:
             An array of sample values for the peak of each spike.

    """

    spikes, peak_samples = [], []
    size = patch_size/2
    for peak in peaks:
        start = peak - size if (peak-size) > 0 else 0
        end = peak+size
        patch = data[start:end]
        if polarity == 'pos':
                peak_sample = patch.argmax()
        elif polarity == 'neg':
            peak_sample = patch.argmin()
        centered = start+peak_sample+offset
        peak_sample = start+peak_sample
        final_patch = data[centered-size:centered+size]
        peak_samples.append(peak_sample)
        # Padding the final patch to patch_size
        final_patch = np.pad(final_patch, (0, patch_size-len(final_patch)),
                             mode='constant', constant_values=0)
        spikes.append(final_patch)

    return np.array(spikes), np.array(peak_samples)