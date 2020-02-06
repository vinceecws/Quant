import numpy as np
from .Indicators import Indicators
from .ExponentialMovingAverage import ExponentialMovingAverage

class PercentageVolumeOscillator(Indicators):
    _name = 'PercentageVolumeOscillator'

    def __init__(self, ohlcv, period=(12, 26, 9)):
        if (period[0] >= period[1]):
            raise ValueError('Short EMA period must be < Long EMA period')

        super(PercentageVolumeOscillator, self).__init__(ohlcv, period)
        self._short_period = self._period[0]
        self._long_period = self._period[1]
        self._signal_period = self._period[2]
        self._short_ema = ExponentialMovingAverage(self._ohlcv, self._short_period, col=4)
        self._long_ema = ExponentialMovingAverage(self._ohlcv, self._long_period, col=4)
        self.generateData()

    def __str__(self):
        return self._name + ' ' + str(self._period)

    @property
    def short_period(self):
        return self._short_period
    
    @property
    def long_period(self):
        return self._long_period
    
    @property
    def signal_period(self):
        return self._signal_period

    @property
    def short_ema(self):
        return self._short_ema

    @property
    def long_ema(self):
        return self._long_ema
    
    @property
    def pvo(self):
        return self._pvo
    
    @property
    def signal(self):
        return self._signal.ema
    
    @property
    def pvo_histogram(self):
        return self._pvo_histogram

    '''
    The Percentage Volume Oscillator (PVO) is a momentum oscillator for volume. 
    The PVO measures the difference between two volume-based moving averages as a percentage of the larger moving average.

    Output(s):
    Percentage Volume Oscillator: ((12-day EMA of Volume - 26-day EMA of Volume)/26-day EMA of Volume) x 100
    Signal Line: 9-day EMA of PVO
    Percentage Volume Oscillator Histogram: PVO - Signal Line
    '''
    
    def generateData(self):

        self._pvo = np.zeros((self._long_ema.firstYieldIndex(), 1), dtype=self._ohlcv.dtype)

        for i in np.arange(self._long_ema.firstYieldIndex(), self._ohlcv.shape[0]):
            short_ema = self._short_ema.ema[i, 0]
            long_ema = self._long_ema.ema[i, 0]
            val = ((short_ema - long_ema) / long_ema) * 100
            self._pvo = np.append(self._pvo, [[val]], axis=0)

        self._signal = ExponentialMovingAverage(self._pvo, self._signal_period, col=0)

        self._pvo_histogram = self._pvo - self._signal.ema

    def firstYieldIndex(self):
        return max(self._long_ema.firstYieldIndex(), self._signal.firstYieldIndex())

    def aboveThreshold(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._pvo_histogram[bar, 0] > 0.0

    def getValue(self, bar):
        return self.aboveThreshold(bar)

    def checkPeriod(self):
        if (self._ohlcv.shape[0] < max(self._period)):
            raise ValueError('Data given has period of {}, while desired periods are {}'.format(self._ohlcv.shape[0], self._period))
