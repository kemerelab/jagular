__all__ = ['JagularModel']

import numpy as np
import copy
import numbers

from functools import wraps
from scipy import interpolate
from sys import float_info
from collections import namedtuple

class JagularModel(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """index access"""
        return None
