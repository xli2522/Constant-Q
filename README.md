[![PyPI version](https://badge.fury.io/py/constantQ.svg)](https://badge.fury.io/py/constantQ)
## A Constant Q Transform based on GWpy qtransform

The creation of this program was inspired by the need to include a CQT package with minimal size and dependency for SHARCNET (ComputeCanada) Supercomputer Clusters.

```None
# IMPORTANT DISCLAIMER: All credits for the original Q transform algorithm go to the authors of *GWpy* and *Omega* pipeline.
# See original algorithms at: [Omega Scan] https://gwdetchar.readthedocs.io/en/stable/omega/
#                             [GWpy] https://gwpy.github.io/docs/stable/
#          particularly       [GWpy qtransform]
#              - https://github.com/gwpy/gwpy/blob/26f63684db17104c5d552c30cdf01248b2ec76c9/gwpy/signal/qtransform.py
#
# The license information does NOT imply this package (constantQ) as the original q transform/q scan algorithm.
# NOTE: Referenced programs are under the GNU license 
# for more information on the license visit: https://www.gnu.org/licenses/gpl-faq.en.html
```

How to use it:

Step 1: Generating a chirp signal

```Python
import numpy as np

# Generate np.array chirp signal
dt = 0.001
t = np.arange(0,3,dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2*np.pi*t*(f0 + (f1 - f0)*np.power(t, 2)/(3*t1**2)))
fs = 1/dt

plt.plot(x)				# plot the chirp signal
plt.show()				# display
```

Step 2: Generating a TimeSeries object

```Python
from constantQ.timeseries import TimeSeries
series = TimeSeries(x, dt = 0.001, unit='m', name='test', t0=0)     #np.array --> constantQ.timeseries    
```

Step 3: Q Transform

```Python
hdata = series
sq = hdata.q_transform(search=None)				# q transform
print(len(sq[0]))       # freq array length
print(len(sq))          # time array length

plt.imshow(sq.T, origin='lower')				# plot the spectrogram
plt.colorbar()									# colorbar
plt.show()										# display
```

To compare the result with a Scipy Spectrogram

```Python
from scipy import signal as scisignal

freq, ts, Sxx = scisignal.spectrogram(x)		# scipy spectrogram

plt.pcolor(ts, freq, Sxx, shading='auto')		# plot the spectrogram
plt.colorbar()									# colorbar
plt.show()										# display
```

This test version 0.0.1 largely follows the GWpy architecture. Changes will be made in future updates if a different structure is better for this package.

