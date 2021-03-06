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
__author__ = 'X. Li'

import numpy as np
import matplotlib.pyplot as plt
from constantQ.timeseries import TimeSeries
import time

# Generate np.array chirp signal
dt = 0.001
t = np.arange(0,3,dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2*np.pi*t*(f0 + (f1 - f0)*np.power(t, 2)/(3*t1**2)))
fs = 1/dt

# Constant Q Transform - not properly labeled 
series = TimeSeries(x, dt = 0.001, unit='m', name='test', t0=0)     #np.array --> gwpy.timeseries    
hdata = series
dstTime = time.time()
sq = hdata.q_transform(search=None)
print(len(sq[0]))       # freq N
print(len(sq))          # time N

current = time.time()
print('DST Time: '+str(current - dstTime))
plt.imshow(sq.T, origin='lower')
#plt.pcolor(sq.T)
plt.colorbar()
plt.show()

# Discrete Time Fourier Transform - not properly labeled 
from scipy import signal as scisignal
dtftTime = time.time()
freq, ts, Sxx = scisignal.spectrogram(x)
print('DTFT Time: '+str(time.time() - dtftTime))

plt.pcolor(ts, freq, Sxx, shading='auto')
plt.colorbar()
plt.show()