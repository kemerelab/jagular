"""This module contains helper functions and utilities for jagular."""

__all__ = ['frange',
           'pairwise',
           'is_sorted',
           'PrettyDuration'
          ]

from numpy import isinf, linspace
from itertools import tee
from collections import namedtuple
from math import floor

def frange(start, stop, step):
    """arange with floating point step"""
    # TODO: this function is not very general; we can extend it to work
    # for reverse (stop < start), empty, and default args, etc.
    num_steps = floor((stop-start)/step)
    return linspace(start, stop, num=num_steps, endpoint=False)

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
        if isinf(seconds):
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
