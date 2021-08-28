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

from constantQ.series import (Array2D, Series)
from constantQ.timeseries import TimeSeries
from constantQ.frequencyseries import FrequencySeries

################### Spectrogram ###################
class Spectrogram(Array2D):
    """A 2D array holding a spectrogram of time-frequency data
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/spectrogram/spectrogram.py
    """
    _metadata_slots = Series._metadata_slots + ('y0', 'dy', 'yindex')
    _default_xunit = TimeSeries._default_xunit
    _default_yunit = FrequencySeries._default_xunit
    _rowclass = TimeSeries
    _columnclass = FrequencySeries

    def __new__(cls, data, unit=None, t0=None, dt=None, f0=None, df=None,
                times=None, frequencies=None,
                name=None, channel=None, **kwargs):
        """Generate a new Spectrogram.
        """
        # parse t0 or epoch
        epoch = kwargs.pop('epoch', None)
        if epoch is not None and t0 is not None:
            raise ValueError("give only one of epoch or t0")
        if epoch is None and t0 is not None:
            #kwargs['x0'] = _format_time(t0)
            pass
        elif epoch is not None:
            #kwargs['x0'] = _format_time(epoch)
            pass
        # parse sample_rate or dt
        if dt is not None:
            kwargs['dx'] = dt
        # parse times
        if times is not None:
            kwargs['xindex'] = times

        # parse y-axis params
        if f0 is not None:
            kwargs['y0'] = f0
        if df is not None:
            kwargs['dy'] = df
        if frequencies is not None:
            kwargs['yindex'] = frequencies

        # generate Spectrogram
        return super().__new__(cls, data, unit=unit, name=name,
                               channel=channel, **kwargs)

    # -- Spectrogram properties -----------------

    epoch = property(TimeSeries.epoch.__get__, TimeSeries.epoch.__set__,
                     TimeSeries.epoch.__delete__,
                     """Starting GPS epoch for this `Spectrogram`
                     :type: `~gwpy.segments.Segment`
                     """)

    t0 = property(TimeSeries.t0.__get__, TimeSeries.t0.__set__,
                  TimeSeries.t0.__delete__,
                  """GPS time of first time bin
                  :type: `~astropy.units.Quantity` in seconds
                  """)

    dt = property(TimeSeries.dt.__get__, TimeSeries.dt.__set__,
                  TimeSeries.dt.__delete__,
                  """Time-spacing for this `Spectrogram`
                  :type: `~astropy.units.Quantity` in seconds
                  """)

    span = property(TimeSeries.span.__get__, TimeSeries.span.__set__,
                    TimeSeries.span.__delete__,
                    """GPS [start, stop) span for this `Spectrogram`
                    :type: `~gwpy.segments.Segment`
                    """)

    f0 = property(Array2D.y0.__get__, Array2D.y0.__set__,
                  Array2D.y0.__delete__,
                  """Starting frequency for this `Spectrogram`
                  :type: `~astropy.units.Quantity` in Hertz
                  """)

    df = property(Array2D.dy.__get__, Array2D.dy.__set__,
                  Array2D.dy.__delete__,
                  """Frequency spacing of this `Spectrogram`
                  :type: `~astropy.units.Quantity` in Hertz
                  """)

    times = property(fget=Array2D.xindex.__get__,
                     fset=Array2D.xindex.__set__,
                     fdel=Array2D.xindex.__delete__,
                     doc="""Series of GPS times for each sample""")

    frequencies = property(fget=Array2D.yindex.__get__,
                           fset=Array2D.yindex.__set__,
                           fdel=Array2D.yindex.__delete__,
                           doc="Series of frequencies for this Spectrogram")

    band = property(fget=Array2D.yspan.__get__,
                    fset=Array2D.yspan.__set__,
                    fdel=Array2D.yspan.__delete__,
                    doc="""Frequency band described by this `Spectrogram`""")
