"""This module contains helper functions and utilities for jagular."""

__all__ = ['frange',
           'pairwise',
           'is_sorted',
           'PrettyDuration'
          ]

import copy
import numpy as np

from itertools import tee
from collections import namedtuple
from math import floor
from warnings import warn

def frange(start, stop, step):
    """arange with floating point step"""
    # TODO: this function is not very general; we can extend it to work
    # for reverse (stop < start), empty, and default args, etc.
    num_steps = floor((stop-start)/step)
    return np.linspace(start, stop, num=num_steps, endpoint=False)

def pairwise(iterable):
    """returns a zip of all neighboring pairs.
    This is used as a helper function for is_sorted.

    Example
    -------
    >>> mylist = [2, 3, 6, 8, 7]
    >>> list(pairwise(mylist))
    [(2, 3), (3, 6), (6, 8), (8, 7)]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def is_sorted_old(iterable, key=lambda a, b: a <= b):
    """Returns True if iterable is monotonic increasing (sorted).

    This function works out-of-core with time complexity O(N), and a very modest
    memory footprint. TODO: does the all key actually quit early, or does it
    require all elements to be compared? And what about memory footprint? Answer
    it should quit early, with no real memory footprint, since all() is
    equivalent to
        def all(iterable):
            for element in iterable:
            if not element:
                return False
        return True
    """
    return all(key(a, b) for a, b in pairwise(iterable))

def is_sorted(x, chunk_size=None):
    """Returns True if iterable is monotonic increasing (sorted).

    NOTE: intended for 1D array.

    This function works in-core with memory footrpint XXX.
    chunk_size = 100000 is probably a good choice.
    """
    
    if isinstance(x, np.ndarray):
        if chunk_size is None:
            chunk_size = 500000
        stop = x.size
        for chunk_start in range(0, stop, chunk_size):
            chunk_stop = int(min(stop, chunk_start + chunk_size + 1))
            chunk = x[chunk_start:chunk_stop]
            if not np.all(chunk[:-1] <= chunk[1:]):
                return False
        return True
    else:
        return is_sorted_old(x)

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def has_duplicate_timestamps(timestamps, *, assume_sorted=None, in_core=True):
    """Docstring goes here.

    WARNING! THIS FUNCTION ASSUMES INTEGRAL TIMESTAMPS!
    """
    if not assume_sorted:
        if not is_sorted(timestamps):
            timestamps = np.sort(timestamps)

    if in_core:
        if np.any(np.diff(timestamps)<1):
            return True
    else:
        raise NotImplementedError("out-of-core still needs to be implemented!")
    return False

def get_duplicate_timestamps(timestamps, *, assume_sorted=None, in_core=True):
    """Docstring goes here.
    Important! Returns indices of duplicate timestamps, not timestamps directly.
    For example, if the timestamps are [0, 1, 2, 10, 11, 11, 11, 12] then this
    function will return np.array([5, 6])

    WARNING! THIS FUNCTION ASSUMES INTEGRAL TIMESTAMPS!
    """
    if not assume_sorted:
        if not is_sorted(timestamps):
            timestamps = np.sort(timestamps)
    duplicates = []
    if in_core:
        if np.any(np.diff(timestamps)<1):
            duplicates = np.atleast_1d((np.argwhere(np.diff(timestamps)<1)+1).squeeze())
    else:
        raise NotImplementedError("out-of-core still needs to be implemented!")
    return duplicates

def get_gap_lengths_from_timestamps(timestamps, *, assume_sorted=None,
                                    in_core=True):
    """Docstring goes here."""
    cs = get_contiguous_segments(data=timestamps,
                                 assume_sorted=assume_sorted,
                                 in_core=in_core)
    gap_lengths = cs[1:,0] - cs[:-1,1]
    return gap_lengths

def get_contiguous_segments(data, *, step=None, assume_sorted=None,
                            in_core=True, index=False, inclusive=False):
    """Compute contiguous segments (separated by step) in a list.

    Note! This function requires that a sorted list is passed.
    It first checks if the list is sorted O(n), and only sorts O(n log(n))
    if necessary. But if you know that the list is already sorted,
    you can pass assume_sorted=True, in which case it will skip
    the O(n) check.

    Returns an array of size (n_segments, 2), with each row
    being of the form ([start, stop]) [inclusive, exclusive].

    NOTE: when possible, use assume_sorted=True, and step=1 as explicit
          arguments to function call.

    WARNING! Step is robustly computed in-core (i.e., when in_core is
        True), but is assumed to be 1 when out-of-core.

    Example
    -------
    >>> data = [1,2,3,4,10,11,12]
    >>> get_contiguous_segments(data)
    ([1,5], [10,13])
    >>> get_contiguous_segments(data, index=True)
    ([0,4], [4,7])

    Parameters
    ----------
    data : array-like
        1D array of sequential data, typically assumed to be integral (sample
        numbers).
    step : float, optional
        Expected step size for neighboring samples. Default uses numpy to find
        the median, but it is much faster and memory efficient to explicitly
        pass in step=1.
    assume_sorted : bool, optional
        If assume_sorted == True, then data is not inspected or re-ordered. This
        can be significantly faster, especially for out-of-core computation, but
        it should only be used when you are confident that the data is indeed
        sorted, otherwise the results from get_contiguous_segments will not be
        reliable.
    in_core : bool, optional
        If True, then we use np.diff which requires all the data to fit
        into memory simultaneously, otherwise we use groupby, which uses
        a generator to process potentially much larger chunks of data,
        but also much slower.
    index : bool, optional
        If True, the indices of segment boundaries will be returned. Otherwise,
        the segment boundaries will be returned in terms of the data itself.
        Default is False.
    inclusive : bool, optional
        If True, the boundaries are returned as [(inclusive idx, inclusive idx)]
        Default is False, and can only be used when index==True.
    """

    if inclusive:
        assert index, "option 'inclusive' can only be used with 'index=True'"
    if in_core:
        data = np.asarray(data)

        if not assume_sorted:
            if not is_sorted(data):
                data = np.sort(data)  # algorithm assumes sorted list

        if step is None:
            step = np.median(np.diff(data))

        # assuming that data(t1) is sampled somewhere on [t, t+1/fs) we have a 'continuous' signal as long as
        # data(t2 = t1+1/fs) is sampled somewhere on [t+1/fs, t+2/fs). In the most extreme case, it could happen
        # that t1 = t and t2 = t + 2/fs, i.e. a difference of 2 steps.

        if np.any(np.diff(data) < step):
            warn("some steps in the data are smaller than the requested step size.")

        breaks = np.argwhere(np.diff(data)>=2*step)
        starts = np.insert(breaks+1, 0, 0).astype(int)
        stops = np.append(breaks, len(data)-1).astype(int)
        bdries = np.vstack((data[starts], data[stops] + step)).T
        if index:
            if inclusive:
                indices = np.vstack((starts, stops)).T
            else:
                indices = np.vstack((starts, stops + 1)).T
            return indices
    else:
        from itertools import groupby
        from operator import itemgetter

        if not assume_sorted:
            if not is_sorted(data):
                # data = np.sort(data)  # algorithm assumes sorted list
                raise NotImplementedError("out-of-core sorting has not been implemented yet...")

        if step is None:
            step = 1

        bdries = []

        if not index:
            for k, g in groupby(enumerate(data), lambda ix: (ix[0] - ix[1])):
                f = itemgetter(1)
                gen = (f(x) for x in g)
                start = next(gen)
                stop = start
                for stop in gen:
                    pass
                bdries.append([start, stop + step])
        else:
            counter = 0
            for k, g in groupby(enumerate(data), lambda ix: (ix[0] - ix[1])):
                f = itemgetter(1)
                gen = (f(x) for x in g)
                _ = next(gen)
                start = counter
                stop = start
                for _ in gen:
                    stop +=1
                if inclusive:
                    bdries.append([start, stop])
                else:
                    bdries.append([start, stop + 1])
                counter = stop + 1

    return np.asarray(bdries)

def _get_contiguous_segments_fast(data, *, step=None, assume_sorted=None,
                            index=False, inclusive=False):
    """Compute contiguous segments (separated by step) in a list.

    Note! This function is fast, but is not finalized.
    """

    if inclusive:
        assert index, "option 'inclusive' can only be used with 'index=True'"
    data = np.asarray(data)

    if not assume_sorted:
        if not is_sorted(data):
            data = np.sort(data)  # algorithm assumes sorted list

    if step is None:
        step = np.median(np.diff(data))

    # assuming that data(t1) is sampled somewhere on [t, t+1/fs) we have a 'continuous' signal as long as
    # data(t2 = t1+1/fs) is sampled somewhere on [t+1/fs, t+2/fs). In the most extreme case, it could happen
    # that t1 = t and t2 = t + 2/fs, i.e. a difference of 2 steps.

    chunk_size = 1000000
    stop = data.size
    breaks = []
    for chunk_start in range(0, stop, chunk_size):
        chunk_stop = int(min(stop, chunk_start + chunk_size + 2))
        breaks_in_chunk = chunk_start + np.argwhere(np.diff(data[chunk_start:chunk_stop])>=2*step)
        if np.any(breaks_in_chunk):
            breaks.extend(breaks_in_chunk)
    breaks = np.array(breaks)
    starts = np.insert(breaks+1, 0, 0).astype(int)
    stops = np.append(breaks, len(data)-1).astype(int)
    bdries = np.vstack((data[starts], data[stops] + step)).T 
    if index:
        if inclusive:
            indices = np.vstack((starts, stops)).T
        else:
            indices = np.vstack((starts, stops + 1)).T
        return indices
    return np.asarray(bdries)

def sanitize_timestamps(timestamps, max_gap_size=150, in_core=True, ts_dtype=None, verbose=True):
    """
    max_gap_size: in samples, inclusive, which will be interpolated over
    """

    def is_integer(my_list):
        """
        my_list = [1,2,5,6, 9.0, '65'] # True, since all elements can be cast without loss to integers
        my_list = [1,2,5,6, 9.0, 'a'] # False, since 'a' is not an integer
        """
        try:
            return all(float(item).is_integer() for item in my_list)
        except ValueError:
            pass
        return False

    if ts_dtype is None:
        ts_dtype = np.uint32

    timestamps_new = copy.copy(timestamps)

    # step 1: make sure that timestamps are integral, and the expected datatype:
    if isinstance(timestamps, np.ndarray):
        if timestamps.dtype != ts_dtype:
            raise TypeError('timestamps are in an unexpected format: {} expected, but {} found!'.format(ts_dtype, timestamps.dtype))
    elif isinstance(timestamps, list):
        if not is_integer(timestamps):
            raise TypeError('timestamps are in an unexpected format; integral values expected, but non-integral values found!')
    else:
        raise TypeError('timestamps are in an unexpected format!')

    # step 2: make sure that timestamps are ordered
    if in_core:
        if not is_sorted(timestamps):
            ts = np.sort(timestamps) # this assumes in-core
    else:
        raise NotImplementedError('out-of-core sorting has not been implemented yet')

    # step 3: check for, and remove duplicate timestamps
    dupes_to_drop = []
    dupes = get_duplicate_timestamps(timestamps=timestamps)
    if np.any(dupes):
        if verbose:
            print('{} duplicate timestamp(s) found; only keeping data corresponding to first occurence(s)'.format(len(dupes)))
        #TODO: drop duplicate timestamps and corresponding data
        timestamps_new = np.delete(timestamps, dupes)
        dupes_to_drop = dupes

    # step 4: check for, and prepare for dealing with missing timestamps
    if verbose:
        gap_lengths = get_gap_lengths_from_timestamps(timestamps=timestamps, in_core=True)
        if gap_lengths.sum():
            print('{} samples are missing from interior of current block; {}+ samples will be filled in by interpolation'.format(int(gap_lengths.sum()), int(np.where(gap_lengths < max_gap_size, gap_lengths, 0).sum())))

    return timestamps_new, dupes_to_drop

def check_timestamps(timestamps, ts_dtype=None):
    """Docstring goes here.
    """

    def is_integer(my_list):
        """
        my_list = [1,2,5,6, 9.0, '65'] # True, since all elements can be cast without loss to integers
        my_list = [1,2,5,6, 9.0, 'a'] # False, since 'a' is not an integer
        """
        try:
            return all(float(item).is_integer() for item in my_list)
        except ValueError:
            pass
        return False

    if ts_dtype is None:
        ts_dtype = np.uint32

    # step 1: make sure that timestamps are integral, and the expected datatype:
    if isinstance(timestamps, np.ndarray):
        if timestamps.dtype != ts_dtype:
            print('timestamps are in an unexpected format: {} expected, but {} found!'.format(ts_dtype, timestamps.dtype))
            return False
    elif isinstance(timestamps, list):
        if not is_integer(timestamps):
            print('timestamps are in an unexpected format; integral values expected, but non-integral values found!')
            return False
    else:
        print('timestamps are in an unexpected format!')
        return False

    # step 2: make sure that timestamps are ordered
    if not is_sorted(timestamps):
        print('timestamps are not sorted in increasing order')
        return False

    # step 3: check for duplicate timestamps
    dupes = get_duplicate_timestamps(timestamps=timestamps)
    if dupes:
        print('{} duplicate timestamp(s) found'.format(len(dupes)))
        return False

    return True

def extract_channels(jfm,*, ts_out=None, max_gap_size=None, ch_out_prefix=None, subset='all',
                     block_size=None, ts_dtype=None, verbose=False, **kwargs):
    """Docstring goes here

    Parameters
    ==========
    jfm: JagularFileMap
    ts_out: string, optional
        Filename of timestamps file; defaults to 'timestamps.raw'
    max_gap_size: int, optional
        Number of samples (inclusive) to fill with linear interpolation.
        Default is 0.
    ch_out_prefix: string, optional
        Prefix to append to filename: prefixch.xx.raw. Default is None.
    subset: string or array-like, optional
        List of channels to write out, default is 'all'.
    block_size: int, optional
        Number of packets to read in at a time. Default is 65536
    ts_dtype: np.dtype, optional
        Type for timestamps, default is np.uint32.
        NOTE: currently no other types are supported!

    Returns
    =======
        None

    TODO: add format options for both channel data, and timestamp data!
    TODO: check whether timestamps.raw or whatever already exists

    """
    from contextlib import ExitStack
    from scipy.interpolate import interp1d
    from struct import Struct

    if ts_out is None:
        ts_out = 'timestamps.raw'
    if max_gap_size is None:
        max_gap_size = 0
    if subset == 'all':
        subset = range(jfm._reader.n_spike_channels)
    if block_size is None:
        block_size = 65536
    if ts_dtype is None:
        ts_dtype = np.uint32
    else:
        raise NotImplementedError('Only np.uint32 is currently supported for ts_dtype!')

    n_chan_zfill = len(str(jfm._reader.n_spike_channels))

    ch_out_files = [ch_out_prefix + 'ch.' + str(n).zfill(n_chan_zfill) + '.raw' for n in subset]

    prev_channel_data = None # used for across-block interpolation
    prev_ts_data = None      # used for across-block interpolation
    # assumption: block_size >> interp_size (we can check for this with an assert), but actually works
    # even when this assumption is not satisfied... yeah!!!

    with ExitStack() as stack:
        ts_file = stack.enter_context(open(ts_out, 'wb+'))
        ch_files = [stack.enter_context(open(fname, 'wb+')) for fname in ch_out_files]

        for ii, (ts, all_ch_data) in enumerate(jfm.read_stitched_files(block_size=block_size)):
            if verbose:
                print('processing block {}'.format(ii))

            ts, dupes_to_drop = sanitize_timestamps(ts, verbose=verbose)
            all_ch_data = np.delete(all_ch_data, dupes_to_drop, axis=1)

            if max_gap_size > 0:

                if prev_ts_data is not None:
                    inter_block_gap = ts[0] - prev_ts_data
                    if (inter_block_gap <= max_gap_size) & (inter_block_gap > 1):
                        print('we need to interpolate across blocks! (block {} to {}, sample {} to {})'.format(ii-1, ii, prev_ts_data, ts[0]))
                        pre_ts = np.arange(prev_ts_data, ts[0])
                        f = interp1d([prev_ts_data, ts[0]], np.vstack([prev_channel_data, all_ch_data[:,0]]).T, assume_sorted=True)
                        pre_ch = f(pre_ts) # in floats, not np.int16!
                        pre_ch = pre_ch.astype(np.int16) # FB! TODO: make this argument dependent!
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
                step = kwargs.get('step', None)
                cs = get_contiguous_segments(ts, assume_sorted=True, step=step).astype(ts_dtype)
                gap_lengths = cs[1:,0] - cs[:-1,1]

                if np.any(gap_lengths <= max_gap_size):
                    # only do this if there are some gaps satisfying the criteria
                    tt = np.argwhere(gap_lengths<=max_gap_size)
                    vv = np.argwhere(gap_lengths>max_gap_size)
                    ccl = (np.cumsum(cs[:,1] - cs[:,0]) - 1).astype(ts_dtype)
                    ccr = np.cumsum(cs[:,1] - cs[:,0]).astype(ts_dtype)
                    orig_data_locs = np.vstack((np.insert(ccr[:-1],0,0),ccr)).T # want this as separate function, too!
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
            my_ch_struct = Struct('<%dh' % num_packets)
            my_ts_struct = Struct('<%dI' % num_packets) # ts_dtype should affect this!!!!
            ts_packed = my_ts_struct.pack(*ts)

            for ii, ch in enumerate(subset):
                ch_packed = my_ch_struct.pack(*all_ch_data[ch,:])
                # write current channel data of current block to file:
                ch_files[ii].write(ch_packed)

            # write timestamps of current block to file:
            ts_file.write(ts_packed)

    # inspect entire timestamps file to check for consistency:
    ts = np.fromfile(ts_out, dtype=ts_dtype)
    if not check_timestamps(ts):
        raise ValueError('block-level timestamps were consistent, but session-level timestamps still have errors!')
    if verbose:
        print('all timestamps OK')


class PrettyBytes(int):
    """Prints number of bytes in a more readable format"""

    def __init__(self, val):
        self.val = val

    def __str__(self):
        if self.val < 1024:
            return '{} bytes'.format(self.val)
        elif self.val < 1024**2:
            return '{:.3f} kilobytes'.format(self.val/1024)
        elif self.val < 1024**3:
            return '{:.3f} megabytes'.format(self.val/1024**2)
        elif self.val < 1024**4:
            return '{:.3f} gigabytes'.format(self.val/1024**3)

    def __repr__(self):
        return self.__str__()

class PrettyInt(int):
    """Prints integers in a more readable format"""

    def __init__(self, val):
        self.val = val

    def __str__(self):
        return '{:,}'.format(self.val)

    def __repr__(self):
        return '{:,}'.format(self.val)

class PrettyDuration(float):
    """Time duration with pretty print.

    Behaves like a float, and can always be cast to a float.
    """

    def __init__(self, seconds):
        self.duration = seconds

    def __str__(self):
        return self.time_string(self.duration)

    def __repr__(self):
        return self.time_string(self.duration)

    @staticmethod
    def to_dhms(seconds):
        """convert seconds into hh:mm:ss:ms"""
        pos = seconds >= 0
        if not pos:
            seconds = -seconds
        ms = seconds % 1; ms = round(ms*10000)/10
        seconds = floor(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        Time = namedtuple('Time', 'pos dd hh mm ss ms')
        time = Time(pos=pos, dd=d, hh=h, mm=m, ss=s, ms=ms)
        return time

    @staticmethod
    def time_string(seconds):
        """returns a formatted time string."""
        if np.isinf(seconds):
            return 'inf'
        pos, dd, hh, mm, ss, s = PrettyDuration.to_dhms(seconds)
        if s > 0:
            if mm == 0:
                # in this case, represent milliseconds in terms of
                # seconds (i.e. a decimal)
                sstr = str(s/1000).lstrip('0')
                # if sstr == "1.0":
                #     ss += 1
                #     sstr = ""
            else:
                # for all other cases, milliseconds will be represented
                # as an integer
                sstr = ":{:03d}".format(int(s))
        else:
            sstr = ""
        if dd > 0:
            daystr = "{:01d} days ".format(int(dd))
        else:
            daystr = ""
        if hh > 0:
            timestr = daystr + "{:01d}:{:02d}:{:02d}{} hours".format(hh, mm, ss, sstr)
        elif mm > 0:
            timestr = daystr + "{:01d}:{:02d}{} minutes".format(mm, ss, sstr)
        elif ss > 0:
            timestr = daystr + "{:01d}{} seconds".format(ss, sstr)
        else:
            timestr = daystr +"{} milliseconds".format(s)
        if not pos:
            timestr = "-" + timestr
        return timestr

    def __add__(self, other):
        """a + b"""
        return PrettyDuration(self.duration + other)

    def __radd__(self, other):
        """b + a"""
        return self.__add__(other)

    def __sub__(self, other):
        """a - b"""
        return PrettyDuration(self.duration - other)

    def __rsub__(self, other):
        """b - a"""
        return other - self.duration

    def __mul__(self, other):
        """a * b"""
        return PrettyDuration(self.duration * other)

    def __rmul__(self, other):
        """b * a"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """a / b"""
        return PrettyDuration(self.duration / other)
