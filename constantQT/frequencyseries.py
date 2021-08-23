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

import numpy
from astropy import units
from constantQT.series import Series

################### FrequencySeries ###################
class FrequencySeries(Series):
    """A data array holding some metadata to represent a frequency series
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/frequencyseries/frequencyseries.py
    """
    _default_xunit = units.Unit('Hz')
    _print_slots = ['f0', 'df', 'epoch', 'name', 'channel']

    def __new__(cls, data, unit=None, f0=None, df=None, frequencies=None,
                name=None, epoch=None, channel=None, **kwargs):
        """Generate a new FrequencySeries.
        """
        if f0 is not None:
            kwargs['x0'] = f0
        if df is not None:
            kwargs['dx'] = df
        if frequencies is not None:
            kwargs['xindex'] = frequencies

        # generate FrequencySeries
        return super().__new__(
            cls, data, unit=unit, name=name, channel=channel,
            epoch=epoch, **kwargs)

    # -- FrequencySeries properties -------------

    f0 = property(Series.x0.__get__, Series.x0.__set__, Series.x0.__delete__,
                  """Starting frequency for this `FrequencySeries`
                  :type: `~astropy.units.Quantity` scalar
                  """)

    df = property(Series.dx.__get__, Series.dx.__set__, Series.dx.__delete__,
                  """Frequency spacing of this `FrequencySeries`
                  :type: `~astropy.units.Quantity` scalar
                  """)

    frequencies = property(fget=Series.xindex.__get__,
                           fset=Series.xindex.__set__,
                           fdel=Series.xindex.__delete__,
                           doc="""Series of frequencies for each sample""")

    def interpolate(self, df):
        """Interpolate this `FrequencySeries` to a new resolution.
        """
        f0 = self.f0.decompose().value
        N = (self.size - 1) * (self.df.decompose().value / df) + 1
        fsamples = numpy.arange(0, numpy.rint(N),
                                dtype=self.real.dtype) * df + f0
        out = type(self)(numpy.interp(fsamples, self.frequencies.value,
                                      self.value))
        out.__array_finalize__(self)
        out.f0 = f0
        out.df = df
        return out