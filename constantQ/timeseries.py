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

import numpy
from numpy import fft as npfft
from scipy import signal
from astropy import units

from constantQ.series import (Series, Segment)
import constantQ.spectral as spectral
from constantQ.utilityaccess import (_fft_length_default, fir_from_transfer, recommended_overlap)
import constantQ.qtransform as qtransform
from constantQ.frequencyseries import FrequencySeries

DEFAULT_FFT_METHOD = None

################### TimeSeriesBase ###################
class TimeSeriesBase(Series):
    """An `Array` with time-domain metadata.
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/timeseries/core.py
    """
    _default_xunit = units.second
    _print_slots = ('t0', 'dt', 'name', 'channel')
    DictClass = None

    def __new__(cls, data, unit=None, t0=None, dt=None, sample_rate=None,
                times=None, channel=None, name=None, **kwargs):
        """Generate a new `TimeSeriesBase`.
        """
        # parse t0 or epoch
        epoch = kwargs.pop('epoch', None)
        if epoch is not None and t0 is not None:
            raise ValueError("give only one of epoch or t0")
        if epoch is None and t0 is not None:
            kwargs['x0'] = t0
            
        elif epoch is not None:
            kwargs['x0'] = epoch
        
        # parse sample_rate or dt
        if sample_rate is not None and dt is not None:
            raise ValueError("give only one of sample_rate or dt")
        if sample_rate is None and dt is not None:
            kwargs['dx'] = dt
        # parse times
        if times is not None:
            kwargs['xindex'] = times

        # generate TimeSeries
        new = super().__new__(cls, data, name=name, unit=unit,
                              channel=channel, **kwargs)

        # manually set sample_rate if given
        if sample_rate is not None:
            new.sample_rate = sample_rate

        return new

    # rename properties from the Series
    t0 = Series.x0
    dt = Series.dx
    span = Series.xspan
    times = Series.xindex

    # -- epoch
    # this gets redefined to attach to the t0 property
    @property
    def epoch(self):
        """GPS epoch for these data.
        This attribute is stored internally by the `t0` attribute
        :type: `~astropy.time.Time`
        """
        try:
            return self.t0
        except AttributeError:
            return None

    @epoch.setter
    def epoch(self, epoch):
        if epoch is None:
            del self.t0
        else:
            try:
                self.t0 = epoch
            except TypeError:
                self.t0 = epoch
    
    # -- sample_rate
    @property
    def sample_rate(self):
        """Data rate for this `TimeSeries` in samples per second (Hertz).
        This attribute is stored internally by the `dx` attribute
        :type: `~astropy.units.Quantity` scalar
        """
        return (1 / self.dt).to('Hertz')

    @sample_rate.setter
    def sample_rate(self, val):
        if val is None:
            del self.dt
            return
        self.dt = (1 / units.Quantity(val, units.Hertz)).to(self.xunit)
        if numpy.isclose(self.dt.value, round(self.dt.value)):
            self.dt = units.Quantity(round(self.dt.value), self.dt.unit)

    # -- duration
    @property
    def duration(self):
        """Duration of this series in seconds
        :type: `~astropy.units.Quantity` scalar
        """
        return units.Quantity(self.span[1] - self.span[0], self.xunit,
                              dtype=float)

################### TimeSeries ###################
class TimeSeries(TimeSeriesBase):
    """A time-domain data array.
        https://github.com/gwpy/gwpy/blob/v2.0.4/gwpy/timeseries/timeseries.py
    """
    def fft(self, nfft=None):

        if nfft is None:
            nfft = self.size
        dft = npfft.rfft(self.value, n=nfft) / nfft
        dft[1:] *= 2.0
        new = FrequencySeries(dft, epoch=self.epoch, unit=self.unit,
                              name=self.name, channel=None )#self.channel)
        try:
            new.frequencies = npfft.rfftfreq(nfft, d=self.dx.value)
        except AttributeError:
            new.frequencies = numpy.arange(new.size) / (nfft * self.dx.value)
        return new

    def average_fft(self, fftlength=None, overlap=0, window=None):
        """Compute the averaged one-dimensional DFT
        """
        
        # format lengths
        if fftlength is None:
            fftlength = self.duration
        if isinstance(fftlength, units.Quantity):
            fftlength = fftlength.value
        nfft = int((fftlength * self.sample_rate).decompose().value)
        noverlap = int((overlap * self.sample_rate).decompose().value)

        navg = divmod(self.size-noverlap, (nfft-noverlap))[0]

        # format window
        if window is None:
            window = 'boxcar'
        if isinstance(window, (str, tuple)):
            win = signal.get_window(window, nfft)
        else:
            win = numpy.asarray(window)
            if len(win.shape) != 1:
                raise ValueError('window must be 1-D')
            elif win.shape[0] != nfft:
                raise ValueError('Window is the wrong size.')
        win = win.astype(self.dtype)
        scaling = 1. / numpy.absolute(win).mean()

        if nfft % 2:
            nfreqs = (nfft + 1) // 2
        else:
            nfreqs = nfft // 2 + 1
            
        from constantQ.spectrogram import Spectrogram
        ffts = Spectrogram(numpy.zeros((navg, nfreqs), dtype=numpy.complex128),
                           channel=self.channel, epoch=self.epoch, f0=0,
                           df=1 / fftlength, dt=1, copy=True)
        # stride through TimeSeries, recording FFTs as columns of Spectrogram
        idx = 0
        for i in range(navg):
            # find step TimeSeries
            idx_end = idx + nfft
            if idx_end > self.size:
                continue
            stepseries = self[idx:idx_end].detrend() * win
            # calculated FFT, weight, and stack
            fft_ = stepseries.fft(nfft=nfft) * scaling
            ffts.value[i, :] = fft_.value
            idx += (nfft - noverlap)
        mean = ffts.mean(0)
        mean.name = self.name
        mean.epoch = self.epoch
        mean.channel = self.channel
        return mean

    def psd(self, fftlength=None, overlap=None, window='hann',
            method=DEFAULT_FFT_METHOD, **kwargs):
        """Calculate the PSD
        """
        # get method
        method_func = spectral.get_method(method)

        # calculate PSD using UI method
        return spectral.psd(self, method_func, fftlength=fftlength,
                            overlap=overlap, window=window, **kwargs)

    def asd(self, fftlength=None, overlap=None, window='hann',
            method=DEFAULT_FFT_METHOD, **kwargs):
        """Calculate the ASD
        """
        return self.psd(method=method, fftlength=fftlength, overlap=overlap,
                        window=window, **kwargs) ** (1/2.)

    def csd(self, other, fftlength=None, overlap=None, window='hann',
            **kwargs):
        """Calculate the CSD
        """
        return spectral.psd(
            (self, other),
            spectral.csd,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs
        )

    def spectrogram(self, stride, fftlength=None, overlap=None, window='hann',
                    method=DEFAULT_FFT_METHOD, nproc=1, **kwargs):
        """Calculate the average power spectrogram
        """
        # get method
        method_func = spectral.get_method(method)

        # calculate PSD using UI method
        return spectral.average_spectrogram(
            self,
            method_func,
            stride,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs
        )
   
    # -- TimeSeries filtering -------------------

    def whiten(self, fftlength=None, overlap=0, method=DEFAULT_FFT_METHOD,
               window='hanning', detrend='constant', asd=None,
               fduration=2, highpass=None, **kwargs):
        """Whiten this `TimeSeries` using inverse spectrum truncation
        For more on inverse spectrum truncation, see arXiv:gr-qc/0509116.
        """
        # compute the ASD
        fftlength = fftlength if fftlength else _fft_length_default(self.dt)
        if asd is None:
            asd = self.asd(fftlength, overlap=overlap,
                           method=method, window=window, **kwargs)
        asd = asd.interpolate(1./self.duration.decompose().value)
        # design whitening filter, with highpass if requested
        ncorner = int(highpass / asd.df.decompose().value) if highpass else 0
        ntaps = int((fduration * self.sample_rate).decompose().value)
        tdw = fir_from_transfer(1/asd.value, ntaps=ntaps,
                                              window=window, ncorner=ncorner)
        # condition the input data and apply the whitening filter
        in_ = self.copy().detrend(detrend)
        out = in_.convolve(tdw, window=window)
        return out * numpy.sqrt(2 * in_.dt.decompose().value)

    def convolve(self, fir, window='hanning'):
        """Convolve this `TimeSeries` with an FIR filter using the
           overlap-save method
        """
        pad = int(numpy.ceil(fir.size/2))
        nfft = min(8*fir.size, self.size)
        # condition the input data
        in_ = self.copy()
        window = signal.get_window(window, fir.size)
        in_.value[:pad] *= window[:pad]
        in_.value[-pad:] *= window[-pad:]
        # if FFT length is long enough, perform only one convolution
        if nfft >= self.size/2:
            conv = signal.fftconvolve(in_.value, fir, mode='same')
        # else use the overlap-save algorithm
        else:
            nstep = nfft - 2*pad
            conv = numpy.zeros(self.size)
            # handle first chunk separately
            conv[:nfft-pad] = signal.fftconvolve(in_.value[:nfft], fir,
                                                 mode='same')[:nfft-pad]
            # process chunks of length nstep
            k = nfft - pad
            while k < self.size - nfft + pad:
                yk = signal.fftconvolve(in_.value[k-pad:k+nstep+pad], fir,
                                        mode='same')
                conv[k:k+yk.size-2*pad] = yk[pad:-pad]
                k += nstep
            # handle last chunk separately
            conv[-nfft+pad:] = signal.fftconvolve(in_.value[-nfft:], fir,
                                                  mode='same')[-nfft+pad:]
        out = type(self)(conv)
        out.__array_finalize__(self)
        return out

    def detrend(self, detrend='constant'):
        """Remove the trend from this `TimeSeries`
        """
        data = signal.detrend(self.value, type=detrend).view(type(self))
        data.__metadata_finalize__(self)
        data._unit = self.unit
        return data

    def q_gram(self,
               qrange=qtransform.DEFAULT_QRANGE,
               frange=qtransform.DEFAULT_FRANGE,
               mismatch=qtransform.DEFAULT_MISMATCH,
               snrthresh=5.5,
               **kwargs):
        """Scan a `TimeSeries` using the multi-Q transform and return an
        `Table` of the most significant tiles
        """
        qscan, _ = qtransform.q_scan(self, mismatch=mismatch, qrange=qrange,
                                     frange=frange, **kwargs)
        qgram = qscan.table(snrthresh=snrthresh)
        return qgram

    def q_transform(self,
                    qrange=qtransform.DEFAULT_QRANGE,
                    frange=qtransform.DEFAULT_FRANGE,
                    gps=None,
                    search=.5,
                    tres="<default>",
                    fres="<default>",
                    logf=False,
                    norm='median',
                    mismatch=qtransform.DEFAULT_MISMATCH,
                    outseg=None,
                    whiten=True,
                    fduration=2,
                    highpass=None,
                    **asd_kw):
        """Scan a `TimeSeries` using the multi-Q transform and return an
        interpolated high-resolution spectrogram
        """  # noqa: E501
        
        # condition data
        if whiten is True:  # generate ASD dynamically
            window = asd_kw.pop('window', 'hann')
            fftlength = asd_kw.pop('fftlength',
                                   _fft_length_default(self.dt))                       
            overlap = asd_kw.pop('overlap', None)
            if overlap is None and fftlength == self.duration.value:
                asd_kw['method'] = DEFAULT_FFT_METHOD
                overlap = 0
            elif overlap is None:
                overlap = recommended_overlap(window) * fftlength
            whiten = self.asd(fftlength, overlap, window=window, **asd_kw)
        if isinstance(whiten, FrequencySeries):
            # apply whitening (with error on division by zero)
            with numpy.errstate(all='raise'):
                data = self.whiten(asd=whiten, fduration=fduration,
                                   highpass=highpass)
        else:
            data = self
        # determine search window
        if gps is None:
            search = None
        elif search is not None:
            search = Segment(gps-search/2, gps+search/2) & self.span
        qgram, _ = qtransform.q_scan(
            data, frange=frange, qrange=qrange, norm=norm,
            mismatch=mismatch, search=search)
        return qgram.interpolate(
            tres=tres, fres=fres, logf=logf, outseg=outseg)