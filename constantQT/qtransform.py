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

""" This portion is largely unchanged. Original comments by GWpy developer(s) - Duncan Macleod are also included.
    https://github.com/gwpy/gwpy/blob/26f63684db17104c5d552c30cdf01248b2ec76c9/gwpy/signal/qtransform.py
"""
import warnings

import numpy
from numpy import fft as npfft
from math import (log, ceil, pi, isinf, exp)

from constantQT.segments import Segment
from constantQT.utilityaccess import (round_to_power)

################### Table (EventTable) related ###################
# in GWpy - https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/table/table.py
# here it is equivalent to astropy.table Table
# Table replaces EventTable in GWpy
from astropy.table import Table

################### main q transform related ###################
# https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/signal/qtransform.py

# main q transform
DEFAULT_MISMATCH = None

# q-transform defaults
DEFAULT_FRANGE = (0, float('inf'))
DEFAULT_MISMATCH = 0.2
DEFAULT_QRANGE = (4, 64)

# -- object class definitions -------------------------------------------------

class QObject(object):
    """Base class for Q-transform objects
    This object exists just to provide basic methods for all other
    Q-transform objects.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, duration, sampling, mismatch=DEFAULT_MISMATCH):
        self.duration = float(duration)
        self.sampling = float(sampling)
        self.mismatch = float(mismatch)

    @property
    def deltam(self):
        """Fractional mismatch between neighbouring tiles
        """
        return 2 * (self.mismatch / 3.) ** (1/2.)

class QBase(QObject):
    """Base class for Q-transform objects with fixed Q
    This class just provides a property for Q-prime = Q / sqrt(11)
    """
    def __init__(self, q, duration, sampling, mismatch=DEFAULT_MISMATCH):
        super().__init__(duration, sampling, mismatch=mismatch)
        self.q = float(q)

    @property
    def qprime(self):
        """Normalized Q `(q/sqrt(11))`
        """
        return self.q / 11**(1/2.)

class QTiling(QObject):
    """Iterable constructor of `QPlane` objects
    """
    def __init__(self, duration, sampling,
                 qrange=DEFAULT_QRANGE,
                 frange=DEFAULT_FRANGE,
                 mismatch=DEFAULT_MISMATCH):
        super().__init__(duration, sampling, mismatch=mismatch)
        self.qrange = (float(qrange[0]), float(qrange[1]))
        self.frange = [float(frange[0]), float(frange[1])]

        qlist = list(self._iter_qs())
        if self.frange[0] == 0:  # set non-zero lower frequency
            self.frange[0] = 50 * max(qlist) / (2 * pi * self.duration)
        maxf = self.sampling / 2 / (1 + 11**(1/2.) / min(qlist))
        if isinf(self.frange[1]):
            self.frange[1] = maxf
        elif self.frange[1] > maxf:  # truncate upper frequency to maximum
            warnings.warn('upper frequency of %.2f is too high for the given '
                          'Q range, resetting to %.2f'
                          % (self.frange[1], maxf))
            self.frange[1] = maxf

    @property
    def qs(self):  # pylint: disable=invalid-name
        """Array of Q values for this `QTiling`
        :type: `numpy.ndarray`
        """
        return numpy.array(list(self._iter_qs()))

    @property
    def whitening_duration(self):
        """The recommended data duration required for whitening
        """
        return max(t.whitening_duration for t in self)

    def _iter_qs(self):
        """Iterate over the Q values
        """
        # work out how many Qs we need
        cumum = log(self.qrange[1] / self.qrange[0]) / 2**(1/2.)
        nplanes = int(max(ceil(cumum / self.deltam), 1))
        dq = cumum / nplanes  # pylint: disable=invalid-name
        for i in range(nplanes):
            yield self.qrange[0] * exp(2**(1/2.) * dq * (i + .5))

    def __iter__(self):
        """Iterate over this `QTiling`
        Yields a `QPlane` at each Q value
        """
        for q in self._iter_qs():
            yield QPlane(q, self.frange, self.duration, self.sampling,
                         mismatch=self.mismatch)

    def transform(self, fseries, **kwargs):
        """Compute the time-frequency plane at fixed Q with the most
        significant tile
        Parameters
        ----------
        fseries : `~gwpy.timeseries.FrequencySeries`
            the complex FFT of a time-series data set
        **kwargs
            other keyword arguments to pass to `QPlane.transform`
        Returns
        -------
        out : `QGram`
            signal energies over the time-frequency plane containing the most
            significant tile
        N : `int`
            estimated number of statistically independent tiles
        See also
        --------
        QPlane.transform
            compute the Q-transform over a single time-frequency plane
        """
        if not numpy.isfinite(fseries).all():
            raise ValueError('Input signal contains non-numerical values')
        weight = 1 + numpy.log10(self.qrange[1]/self.qrange[0]) / numpy.sqrt(2)
        nind, nplanes, peak, result = (0, 0, 0, None)
        # identify the plane with the loudest tile
        for plane in self:
            nplanes += 1
            nind += sum([1 + row.ntiles * row.deltam for row in plane])
            result = plane.transform(fseries, **kwargs)
            if result.peak['energy'] > peak:
                out = result
                peak = out.peak['energy']
        return (out, nind * weight / nplanes)

class QPlane(QBase):
    """Iterable representation of a Q-transform plane
    For a given Q, an array of frequencies can be iterated over, yielding
    a `QTile` each time.
    """
    def __init__(self, q, frange, duration, sampling,
                 mismatch=DEFAULT_MISMATCH):
        super().__init__(q, duration, sampling, mismatch=mismatch)
        self.frange = [float(frange[0]), float(frange[1])]

        if self.frange[0] == 0:  # set non-zero lower frequency
            self.frange[0] = 50 * self.q / (2 * pi * self.duration)
        if isinf(self.frange[1]):  # set non-infinite upper frequency
            self.frange[1] = self.sampling / 2 / (1 + 1/self.qprime)

    def __iter__(self):
        """Iterate over this `QPlane`
        Yields a `QTile` at each frequency
        """
        # for each frequency, yield a QTile
        for freq in self._iter_frequencies():
            yield QTile(self.q, freq, self.duration, self.sampling,
                        mismatch=self.mismatch)

    def _iter_frequencies(self):
        """Iterate over the frequencies of this `QPlane`
        """
        # work out how many frequencies we need
        minf, maxf = self.frange
        fcum_mismatch = log(maxf / minf) * (2 + self.q**2)**(1/2.) / 2.
        nfreq = int(max(1, ceil(fcum_mismatch / self.deltam)))
        fstep = fcum_mismatch / nfreq
        fstepmin = 1 / self.duration
        # for each frequency, yield a QTile
        last = None
        for i in range(nfreq):
            this = (
                minf * exp(2 / (2 + self.q**2)**(1/2.) * (i + .5) * fstep)
                // fstepmin * fstepmin
            )
            if this != last:  # yield only unique elements
                yield this
            last = this

    @property
    def frequencies(self):
        """Array of central frequencies for this `QPlane`
        """
        return numpy.array(list(self._iter_frequencies()))

    @property
    def farray(self):
        """Array of frequencies for the lower-edge of each frequency bin
        """
        bandwidths = 2 * pi ** (1/2.) * self.frequencies / self.q
        return self.frequencies - bandwidths / 2.

    @property
    def whitening_duration(self):
        """The recommended data duration required for whitening
        """
        return round_to_power(self.q / (2 * self.frange[0]),
                              base=2, which=None)

    def transform(self, fseries, norm=True, epoch=None, search=None):
        """Calculate the energy `TimeSeries` for the given `fseries`
        """
        out = []
        for qtile in self:
            # get energy from transform
            out.append(qtile.transform(fseries, norm=norm, epoch=epoch))
        return QGram(self, out, search)

class QTile(QBase):
    """Representation of a tile with fixed Q and frequency
    """
    def __init__(self, q, frequency, duration, sampling,
                 mismatch=DEFAULT_MISMATCH):
        super().__init__(q, duration, sampling, mismatch=mismatch)
        self.frequency = frequency

    @property
    def bandwidth(self):
        """The bandwidth for tiles in this row
        """
        return 2 * numpy.pi ** (1/2.) * self.frequency / self.q

    @property
    def ntiles(self):
        """The number of tiles in this row
        """
        tcum_mismatch = self.duration * 2 * numpy.pi * self.frequency / self.q
        return round_to_power(tcum_mismatch / self.deltam,
                              base=2, which='upper')

    @property
    def windowsize(self):
        """The size of the frequency-domain window for this row
        """
        return 2 * int(self.frequency / self.qprime * self.duration) + 1

    def _get_indices(self):
        half = int((self.windowsize - 1) / 2)
        return numpy.arange(-half, half + 1)

    def get_window(self):
        """Generate the bi-square window for this row
        """
        # real frequencies
        wfrequencies = self._get_indices() / self.duration
        # dimensionless frequencies
        xfrequencies = wfrequencies * self.qprime / self.frequency
        # normalize and generate bi-square window
        norm = self.ntiles / (self.duration * self.sampling) * (
            315 * self.qprime / (128 * self.frequency)) ** (1/2.)
        return (1 - xfrequencies ** 2) ** 2 * norm

    def get_data_indices(self):
        """Returns the index array of interesting frequencies for this row
        """
        return numpy.round(
            self._get_indices() + 1 + self.frequency * self.duration,
        ).astype(int)

    @property
    def padding(self):
        """The `(left, right)` padding required for the IFFT
        """
        pad = self.ntiles - self.windowsize
        return (int((pad - 1)/2.), int((pad + 1)/2.))

    def transform(self, fseries, norm=True, epoch=None):
        """Calculate the energy `TimeSeries` for the given fseries
        """

        windowed = fseries[self.get_data_indices()] * self.get_window()
        # pad data, move negative frequencies to the end, and IFFT
        padded = numpy.pad(windowed, self.padding, mode='constant')
        wenergy = npfft.ifftshift(padded)
        # return a `TimeSeries`
        if epoch is None:
            epoch = fseries.epoch
        tdenergy = npfft.ifft(wenergy)

        from constantQT.timeseries import TimeSeries
        cenergy = TimeSeries(tdenergy, x0=epoch,
                             dx=self.duration/tdenergy.size, copy=False)
        energy = type(cenergy)(
            cenergy.value.real ** 2. + cenergy.value.imag ** 2.,
            x0=cenergy.x0, dx=cenergy.dx, copy=False)

        if norm:
            norm = norm.lower() if isinstance(norm, str) else norm
            if norm in (True, 'median'):
                narray = energy / energy.median()
            elif norm in ('mean',):
                narray = energy / energy.mean()
            else:
                raise ValueError("Invalid normalisation %r" % norm)
            return narray.astype("float32", casting="same_kind", copy=False)
        return energy

class QGram(object):
    """Store tile energies over an irregularly gridded plane
    """
    def __init__(self, plane, energies, search):
        self.plane = plane
        self.energies = energies
        self.peak = self._find_peak(search)

    def _find_peak(self, search):
        peak = {'energy': 0, 'snr': None, 'time': None, 'frequency': None}
        for freq, energy in zip(self.plane.frequencies, self.energies):
            if search is not None:
                energy = energy.crop(*search)
            maxidx = energy.value.argmax()
            maxe = energy.value[maxidx]
            if maxe > peak['energy']:
                peak.update({
                    'energy': maxe,
                    'snr': (2 * maxe) ** (1/2.),
                    'time': energy.t0.value + energy.dt.value * maxidx,
                    'frequency': freq,
                })
        return peak

    def interpolate(self, tres="<default>", fres="<default>", logf=False,
                    outseg=None):
        """Interpolate this `QGram` over a regularly-gridded spectrogram
        """
        from scipy.interpolate import (interp2d, InterpolatedUnivariateSpline)
        if outseg is None:
            outseg = self.energies[0].span
        frequencies = self.plane.frequencies
        dtype = self.energies[0].dtype
        # build regular Spectrogram from peak-Q data by interpolating each
        # (Q, frequency) `TimeSeries` to have the same time resolution
        if tres == "<default>":
            tres = abs(Segment(outseg)) / 1000.
        xout = numpy.arange(*outseg, step=tres)
        nx = xout.size
        ny = frequencies.size
        
        from constantQT.spectrogram import Spectrogram
        out = Spectrogram(numpy.empty((nx, ny), dtype=dtype),
                          t0=outseg[0], dt=tres, frequencies=frequencies)
        # record Q in output
        out.q = self.plane.q
        # interpolate rows
        for i, row in enumerate(self.energies):
            xrow = numpy.arange(row.x0.value, (row.x0 + row.duration).value,
                                row.dx.value)
            interp = InterpolatedUnivariateSpline(xrow, row.value)
            out[:, i] = interp(xout).astype(dtype, casting="same_kind",
                                            copy=False)
        if fres is None:
            return out
        # interpolate the spectrogram to increase its frequency resolution
        # --- this is done because Duncan doesn't like interpolated images
        #     since they don't support log scaling
        interp = interp2d(xout, frequencies, out.value.T, kind='cubic')
        if not logf:
            if fres == "<default>":
                fres = .5
            outfreq = numpy.arange(
                self.plane.frange[0], self.plane.frange[1], fres,
                dtype=dtype)
        else:
            if fres == "<default>":
                fres = 500
            outfreq = numpy.geomspace(
                self.plane.frange[0],
                self.plane.frange[1],
                num=int(fres),
            )
        new = type(out)(
            interp(xout, outfreq).T.astype(
                dtype, casting="same_kind", copy=False),
            t0=outseg[0], dt=tres, frequencies=outfreq,
        )
        new.q = self.plane.q
        return new

    def table(self, snrthresh=5.5):
        """Represent this `QPlane` as an `Table`
        """
        # get plane properties
        freqs = self.plane.frequencies
        bws = 2 * (freqs - self.plane.farray)
        # collect table data as a recarray
        names = ('time', 'frequency', 'duration', 'bandwidth', 'energy')
        rec = numpy.recarray((0,), names=names, formats=['f8'] * len(names))
        for f, bw, row in zip(freqs, bws, self.energies):
            ind, = (row.value >= snrthresh ** 2 / 2.).nonzero()
            new = ind.size
            if new > 0:
                rec.resize((rec.size + new,), refcheck=False)
                rec['time'][-new:] = row.times.value[ind]
                rec['frequency'][-new:] = f
                rec['duration'][-new:] = row.dt.to('s').value
                rec['bandwidth'][-new:] = bw
                rec['energy'][-new:] = row.value[ind]
        # save to a table
        out = Table(rec, copy=False)
        out.meta['q'] = self.plane.q
        return out

# q transform [function]
# q_scan
def q_scan(data, mismatch=DEFAULT_MISMATCH, qrange=DEFAULT_QRANGE,
           frange=DEFAULT_FRANGE, duration=None, sampling=None,
           **kwargs):
    """Transform data by scanning over a `QTiling`
    """
    # prepare input
    from constantQT.timeseries import TimeSeries
    if isinstance(data, TimeSeries):
        duration = abs(data.span)
        sampling = data.sample_rate.to('Hz').value
        kwargs.update({'epoch': data.t0.value})
        data = data.fft().value
    # return a raw Q-transform and its significance
    qgram, N = QTiling(duration, sampling, mismatch=mismatch, qrange=qrange,
                       frange=frange).transform(data, **kwargs)
    far = 1.5 * N * numpy.exp(-qgram.peak['energy']) / duration
    return (qgram, far)

