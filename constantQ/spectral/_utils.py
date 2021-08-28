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