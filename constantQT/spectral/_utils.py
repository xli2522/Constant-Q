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

from astropy import units

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

def scale_timeseries_unit(tsunit, scaling='density'):
    """Scale the unit of a `TimeSeries` to match that of a `FrequencySeries`
    Parameters
    ----------
    tsunit : `~astropy.units.UnitBase`
        input unit from `TimeSeries`
    scaling : `str`
        type of frequency series, either 'density' for a PSD, or
        'spectrum' for a power spectrum.
    Returns
    -------
    unit : `~astropy.units.Unit`
        unit to be applied to the resulting `FrequencySeries`.
    """
    # set units
    if scaling == 'density':
        baseunit = units.Hertz
    elif scaling == 'spectrum':
        baseunit = units.dimensionless_unscaled
    else:
        raise ValueError("Unknown scaling: %r" % scaling)
    if tsunit:
        specunit = tsunit ** 2 / baseunit
    else:
        specunit = baseunit ** -1
    return specunit