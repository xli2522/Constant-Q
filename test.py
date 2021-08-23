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
__author__ = 'X. Li'

import numpy as np
import matplotlib.pyplot as plt
from constantQT.timeseries import TimeSeries

# Generate np.array chirp signal
dt = 0.001
t = np.arange(0,3,dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2*np.pi*t*(f0 + (f1 - f0)*np.power(t, 2)/(3*t1**2)))
print(len(x))
fs = 1/dt

series = TimeSeries(x, dt = 1/1000, unit='m', name='test', t0=0)     #np.array --> gwpy.timeseries

plt.plot(series)
#plt.show()

hdata = series
sq = hdata.q_transform(frange=(30, 500),search=None)
print(sq)
print(len(sq[100]))
print(len(sq))
