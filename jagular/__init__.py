"""
Jagular default API.

``jagular`` is an out-of-core pre-processing framework for neuroelectrophysiology
data used in the Kemere lab at Rice University.
"""

# from .subpackage import *

from . import io
from . import utils
# from . import spikedetect
from . import filtering

from . version import __version__