# A Constant Q Transform based on *GWpy qtransform* 
#   The creation of this program was inspired by the need to include a CQT package 
#       with minimal size and dependency for SHARCNET (ComputeCanada) Supercomputer Clusters.
#
# Copyright (C) 2021 Xiyuan Li
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# IMPORTANT DISCLAIMER: All credits for the original Q transform algorithm go to the authors of *GWpy* and *Omega* pipeline.
# See original algorithms at: [Omega Scan] https://gwdetchar.readthedocs.io/en/stable/omega/
#                             [GWpy] https://gwpy.github.io/docs/stable/
#          particularly       [GWpy qtransform]
#              - https://github.com/gwpy/gwpy/blob/26f63684db17104c5d552c30cdf01248b2ec76c9/gwpy/signal/qtransform.py
#
# The license information does NOT imply this package (constantQ) as the original q transform/q scan algorithm.
# NOTE: Referenced programs are under the GNU license 
# 
__version__ = 'Testing 0.0.1'

""" Functions in this file are largely unchanged. Original comments by GWpy developer(s) - Duncan Macleod are also included.
    Source GWpy file is commented above each function.      Detailed comments see: https://github.com/gwpy/gwpy
"""

import numpy

import scipy.signal

from constantQ.frequencyseries import FrequencySeries
from constantQ.spectral._utils import scale_timeseries_unit
import constantQ.spectral._registry as fft_registry



# -- density scaling methods --------------------------------------------------

def _spectral_density(timeseries, segmentlength, noverlap=None, name=None,
                      sdfunc=scipy.signal.welch, **kwargs):
    """Calculate a generic spectral density of this `TimeSeries`
    """
    # compute spectral density
    freqs, psd_ = sdfunc(
        timeseries.value,
        noverlap=noverlap,
        fs=timeseries.sample_rate.decompose().value,
        nperseg=segmentlength,
        **kwargs
    )
    # generate FrequencySeries and return
    unit = scale_timeseries_unit(
        timeseries.unit,
        kwargs.get('scaling', 'density'),
    )
    return FrequencySeries(
        psd_,
        unit=unit,
        frequencies=freqs,
        name=(name or timeseries.name),
        epoch=timeseries.epoch,
        channel=None #timeseries.channel,
    )


def welch(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using Welch's method
    """
    kwargs.setdefault('average', 'mean')
    return _spectral_density(timeseries, segmentlength, **kwargs)


def bartlett(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using Bartlett's method
    """
    kwargs.pop('noverlap', None)
    return _spectral_density(timeseries, segmentlength, noverlap=0, **kwargs)


def median(timeseries, segmentlength, **kwargs):
    """Calculate a PSD using Welch's method with a median average
    """
    kwargs.setdefault('average', 'median')
    return _spectral_density(timeseries, segmentlength, **kwargs)


# register
for func in (welch, bartlett, median):
    fft_registry.register_method(func, name=func.__name__)

    # DEPRECATED:
    fft_registry.register_method(func, name='scipy-{}'.format(func.__name__))


# -- others -------------------------------------------------------------------

def rayleigh(timeseries, segmentlength, noverlap=0):
    """Calculate a Rayleigh statistic spectrum
    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        input `TimeSeries` data.
    segmentlength : `int`
        number of samples in single average.
    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.
    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`
    """
    stepsize = segmentlength - noverlap
    if noverlap:
        numsegs = 1 + int((timeseries.size - segmentlength) / float(noverlap))
    else:
        numsegs = int(timeseries.size // segmentlength)
    tmpdata = numpy.ndarray((numsegs, int(segmentlength//2 + 1)))
    for i in range(numsegs):
        tmpdata[i, :] = welch(
            timeseries[i*stepsize:i*stepsize+segmentlength],
            segmentlength)
    std = tmpdata.std(axis=0)
    mean = tmpdata.mean(axis=0)
    return FrequencySeries(std/mean, unit='', copy=False, f0=0,
                           epoch=timeseries.epoch,
                           df=timeseries.sample_rate.value/segmentlength,
                           channel=timeseries.channel,
                           name='Rayleigh spectrum of %s' % timeseries.name)


def csd(timeseries, other, segmentlength, noverlap=None, **kwargs):
    """Calculate the CSD of two `TimeSeries` using Welch's method
    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        time-series of data
    other : `~gwpy.timeseries.TimeSeries`
        time-series of data
    segmentlength : `int`
        number of samples in single average.
    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.
    **kwargs
        other keyword arguments are passed to :meth:`scipy.signal.csd`
    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        average power `FrequencySeries`
    See also
    --------
    scipy.signal.csd
    """
    # calculate CSD
    kwargs.setdefault('y', other.value)
    return _spectral_density(
        timeseries, segmentlength, noverlap=noverlap,
        name=str(timeseries.name)+'---'+str(other.name),
        sdfunc=scipy.signal.csd, **kwargs)