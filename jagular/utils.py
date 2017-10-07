"""This module contains helper functions and utilities for jagular."""

__all__ = ['frange',
           'pairwise',
           'is_sorted',
           'PrettyDuration'
          ]

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

def is_sorted(iterable, key=lambda a, b: a <= b):
    """Check to see if iterable is monotonic increasing (sorted)."""
    return all(key(a, b) for a, b in pairwise(iterable))

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def has_duplicate_timestamps(timestamps, assume_sorted=None, in_core=True):
    """Docstring goes here."""
    if not assume_sorted:
        if not is_sorted(timestamps):
            timestamps = np.sort(timestamps)

    if in_core:
        if np.any(np.diff(timestamps)<1):
            return True
    else:
        raise NotImplementedError("out-of-core still needs to be implemented!")
    return False

def get_duplicate_timestamps(timestamps, assume_sorted=None, in_core=True):
    """Docstring goes here."""
    if not assume_sorted:
        if not is_sorted(timestamps):
            timestamps = np.sort(timestamps)
    duplicates = []
    if in_core:
        if np.any(np.diff(timestamps)<1):
            duplicates = timestamps[np.argwhere(np.diff(timestamps)<1)]
    else:
        raise NotImplementedError("out-of-core still needs to be implemented!")
    return duplicates

def get_gap_lengths_from_timestamps(timestamps, assume_sorted=None, in_core=True):
    """Docstring goes here."""
    cs = get_contiguous_segments(data=timestamps,
                                 assume_sorted=assume_sorted,
                                 in_core=in_core)
    gap_lengths = cs[1:,0] - cs[:-1,1]
    return gap_lengths

def get_contiguous_segments(data, step=None, assume_sorted=None, in_core=True):
    """Compute contiguous segments (seperated by step) in a list.

    Note! This function requires that a sorted list is passed.
    It first checks if the list is sorted O(n), and only sorts O(n log(n))
    if necessary. But if you know that the list is already sorted,
    you can pass assume_sorted=True, in which case it will skip
    the O(n) check.

    Returns an array of size (n_segments, 2), with each row
    being of the form ([start, stop]) [inclusive, exclusive].

    WARNING! Step is robustly computed in-core (i.e., when in_core is
        True), but is assumed to be 1 when out-of-core.

    Parameters
    ----------
    in_core : bool, optional
        If True, then we use np.diff which requires all the data to fit
        into memory simultaneously, otherwise we use groupby, which uses
        a generator to process potentially much larger chunks of data,
        but also much slower.
    """
    data = np.asarray(data)
    if not assume_sorted:
        if not is_sorted(data):
            data = np.sort(data)  # algorithm assumes sorted list

    if in_core:
        if step is None:
            step = np.median(np.diff(data))

        # assuming that data(t1) is sampled somewhere on [t, t+1/fs) we have a 'continuous' signal as long as
        # data(t2 = t1+1/fs) is sampled somewhere on [t+1/fs, t+2/fs). In the most extreme case, it could happen
        # that t1 = t and t2 = t + 2/fs, i.e. a difference of 2 steps.

        if np.any(np.diff(data) < step):
            warn("some steps in the data are smaller than the requested step size.")

        breaks = np.argwhere(np.diff(data)>=2*step)
        starts = np.insert(breaks+1, 0, 0)
        stops = np.append(breaks, len(data)-1)
        bdries = np.vstack((data[starts], data[stops] + step)).T
    else:
        from itertools import groupby
        from operator import itemgetter

        if step is None:
            step = 1

        bdries = []

        for k, g in groupby(enumerate(data), lambda ix: (ix[0] - ix[1])):
            f = itemgetter(1)
            gen = (f(x) for x in g)
            start = next(gen)
            stop = start
            for stop in gen:
                pass
            bdries.append([start, stop + step])

    return np.asarray(bdries)

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
