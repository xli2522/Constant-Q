# A Constant Q Transform based on *GWpy qtransform* 
#   The creation of this program was inspired by the need to include a CQT package 
#       with minimal size and dependency for SHARCNET (ComputeCanada) Supercomputer Clusters.
# 
# IMPORTANT: All credits for the original Q transform algorithm go to the authors of *GWpy* and *Omega* pipeline.
# See original algorithms at: [Omega Scan] https://gwdetchar.readthedocs.io/en/stable/omega/
#                             [GWpy] https://gwpy.github.io/docs/stable/
#          particularly       [GWpy qtransform]
#              - https://github.com/gwpy/gwpy/blob/26f63684db17104c5d552c30cdf01248b2ec76c9/gwpy/signal/qtransform.py
# NOTE: Referenced programs are under the GNU license 
# 
__version__ = 'Testing 0.0.1'

""" Functions in this file are largely unchanged. Original comments by GWpy developer(s) - Duncan Macleod are also included.
    Source GWpy file is commented above each function.      Detailed comments see: https://github.com/gwpy/gwpy
"""

import sys
import numpy
import math
from numpy import fft as npfft
from scipy import signal

################### full access utilities ###################
# timeseries
def _fft_length_default(dt):
    '''Choose an appropriate FFT length (in seconds) based on a sample rate
    '''
    return int(max(2, math.ceil(2048 * dt.decompose().value)))

# timeseriescore
def gprint(*values, **kwargs): 
    kwargs.setdefault('file', sys.stdout)
    file_ = kwargs['file']
    print(*values, **kwargs)
    file_.flush()

# array
def if_not_none(func, value):
    """Apply func to value if value is not None
    """
    if value is None:
        return
    return func(value)

#qtransform
def round_to_power(x, base=2, which=None):
    """Round a positive value to the nearest integer power of `base`
    """
    if which == 'lower':
        selector = math.floor
    elif which == 'upper':
        selector = math.ceil
    elif which is not None:
        raise ValueError("'which' argument must be one of 'lower', "
                         "'upper', or None")
    else:
        selector = round
    return type(base)(base ** selector(math.log(x, base)))

################### filterdesign related ###################
# filterdesign
def truncate_transfer(transfer, ncorner=None):
    """Smoothly zero the edges of a frequency domain transfer function
    """
    nsamp = transfer.size
    ncorner = ncorner if ncorner else 0
    out = transfer.copy()
    out[0:ncorner] = 0
    out[ncorner:nsamp] *= planck(nsamp-ncorner, nleft=5, nright=5)
    return out

# filterdesign
def truncate_impulse(impulse, ntaps, window='hanning'):
    """Smoothly truncate a time domain impulse response
    """
    out = impulse.copy()
    trunc_start = int(ntaps / 2)
    trunc_stop = out.size - trunc_start
    window = signal.get_window(window, ntaps)
    out[0:trunc_start] *= window[trunc_start:ntaps]
    out[trunc_stop:out.size] *= window[0:trunc_start]
    out[trunc_start:trunc_stop] = 0
    return out

# filterdesign
def fir_from_transfer(transfer, ntaps, window='hanning', ncorner=None):
    """Design a Type II FIR filter given an arbitrary transfer function
    """
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = npfft.irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    # wrap around and normalise to construct the filter
    out = numpy.roll(impulse, int(ntaps/2 - 1))[0:ntaps]
    return out


################### window related ###################
from math import ceil
from scipy.signal.windows import windows as scipy_windows
from scipy.special import expit

# window
def canonical_name(name):
    """Find the canonical name for the given window in scipy.signal
    """
    if name.lower() == 'planck':  # make sure to handle the Planck window
        return 'planck'
    try:  # use equivalence introduced in scipy 0.16.0
        # pylint: disable=protected-access
        return scipy_windows._win_equiv[name.lower()].__name__
    except KeyError:  # no match
        raise ValueError('no window function in scipy.signal equivalent to %r'
                         % name,)


ROV = {
    'boxcar': 0,
    'bartlett': .5,
    'barthann': .5,
    'blackmanharris': .661,
    'flattop': .8,
    'hann': .5,
    'hamming': .5,
    'nuttall': .656,
    'triang': .5
}

# -- recommended overlap ------------------------------------------------------
# source: http://edoc.mpg.de/395068
# window
def recommended_overlap(name, nfft=None):
    """Returns the recommended fractional overlap for the given window
    """
    try:
        name = canonical_name(name)
    except KeyError as exc:
        raise ValueError(str(exc))
    try:
        rov = ROV[name]
    except KeyError:
        raise ValueError("no recommended overlap for %r window" % name)
    if nfft:
        return int(ceil(nfft * rov))
    return rov

# -- Planck taper window ------------------------------------------------------
# source: https://arxiv.org/abs/1003.2939
# window
def planck(N, nleft=0, nright=0):
    """Return a Planck taper window.
    """
    # construct a Planck taper window
    w = numpy.ones(N)
    if nleft:
        w[0] *= 0
        zleft = numpy.array([nleft * (1./k + 1./(k-nleft))
                            for k in range(1, nleft)])
        w[1:nleft] *= expit(-zleft)
    if nright:
        w[N-1] *= 0
        zright = numpy.array([-nright * (1./(k-nright) + 1./k)
                             for k in range(1, nright)])
        w[N-nright:N-1] *= expit(-zright)
    return w
