import numpy as np
from .Indicators import Indicators

class RelativeVigorIndex(Indicators):
    _name = 'RelativeVigorIndex'

    def __init__(self, ohlcv, period=14):
        super(RelativeVigorIndex, self).__init__(ohlcv, period)
        self.generateData()

    def __str__(self):
        return self._name + ' ' + str(self._period)

    @property
    def rvi(self):
        return self._rvi
    
    @property
    def signal(self):
        return self._signal
    
    '''
    Relative Vigor Index:
    The Relative Vigor Index (RVI) is a technical analysis indicator that measures the strength of a trend 
    by comparing a security's closing price to its trading range and smoothing the results.
    (Exit Indicator)

    Output(s):
    Relative Vigor Index: SMA(n) of RVINumerator / SMA(n) of RVIDenominator
        *Numerator: ((Close_i - Open_i) + (2*(Close_(i-1) - Open_(i-1))) + (2*(Close_(i-2) - Open_(i-2))) + (Close_(i-3) - Open_(i-3))) / 6
        *Denominator: ((High_i - Low_i) + (2*(High_(i-1) - Low_(i-1))) + (2*(High_(i-2) - Low_(i-2))) + (High_(i-3) - Low_(i-3))) / 6
    Signal Line: ((RVI_i) + (2*(RVI_(i-1))) + (2*(RVI(i-2))) + (RVI_(i-3))) / 6
    '''

    def generateData(self):

        #Initialize initial periods
        rvi_num = np.zeros((3, 1), dtype=self._ohlcv.dtype)
        rvi_den = np.zeros((3, 1), dtype=self._ohlcv.dtype)
        self._rvi = np.zeros((self._period + 2, 1), dtype=self._ohlcv.dtype)
        self._signal = np.zeros((self._period + 5, 1), dtype=self._ohlcv.dtype)

        for i in np.arange(3, self._period + 2):
            num = ((self._ohlcv[i, 3] - self._ohlcv[i, 0]) + (2 * (self._ohlcv[i - 1, 3] - self._ohlcv[i - 1, 0])) + (2 * (self._ohlcv[i - 2, 3] - self._ohlcv[i - 2, 0])) + (self._ohlcv[i - 3, 3] - self._ohlcv[i - 3, 0])) / 6
            rvi_num = np.append(rvi_num, [[num]], axis=0) #RVINumerator = ((Close_i - Open_i) + (2*(Close_(i-1) - Open_(i-1))) + (2*(Close_(i-2) - Open_(i-2))) + (Close_(i-3) - Open_(i-3))) / 6
            den = ((self._ohlcv[i, 1] - self._ohlcv[i, 2]) + (2 * (self._ohlcv[i - 1, 1] - self._ohlcv[i - 1, 2])) + (2 * (self._ohlcv[i - 2, 1] - self._ohlcv[i - 2, 2])) + (self._ohlcv[i - 3, 1] - self._ohlcv[i - 3, 2])) / 6
            rvi_den = np.append(rvi_den, [[den]], axis=0) #RVIDenominator = ((High_i - Low_i) + (2*(High_(i-1) - Low_(i-1))) + (2*(High_(i-2) - Low_(i-2))) + (High_(i-3) - Low_(i-3))) / 6

        for i in np.arange(self._period + 2, self._period + 5):
            num = ((self._ohlcv[i, 3] - self._ohlcv[i, 0]) + (2 * (self._ohlcv[i - 1, 3] - self._ohlcv[i - 1, 0])) + (2 * (self._ohlcv[i - 2, 3] - self._ohlcv[i - 2, 0])) + (self._ohlcv[i - 3, 3] - self._ohlcv[i - 3, 0])) / 6
            rvi_num = np.append(rvi_num, [[num]], axis=0) #RVINumerator = ((Close_i - Open_i) + (2*(Close_(i-1) - Open_(i-1))) + (2*(Close_(i-2) - Open_(i-2))) + (Close_(i-3) - Open_(i-3))) / 6
            den = ((self._ohlcv[i, 1] - self._ohlcv[i, 2]) + (2 * (self._ohlcv[i - 1, 1] - self._ohlcv[i - 1, 2])) + (2 * (self._ohlcv[i - 2, 1] - self._ohlcv[i - 2, 2])) + (self._ohlcv[i - 3, 1] - self._ohlcv[i - 3, 2])) / 6
            rvi_den = np.append(rvi_den, [[den]], axis=0) #RVIDenominator = ((High_i - Low_i) + (2*(High_(i-1) - Low_(i-1))) + (2*(High_(i-2) - Low_(i-2))) + (High_(i-3) - Low_(i-3))) / 6
            self._rvi = np.append(self._rvi, [[np.mean(rvi_num[-self._period:]) / np.mean(rvi_den[-self._period:])]], axis=0) #RVI = SMA(n) of RVINumerator / SMA(n) of RVIDenominator

        for i in np.arange(self._period + 5, self._ohlcv.shape[0]):
            num = ((self._ohlcv[i, 3] - self._ohlcv[i, 0]) + (2 * (self._ohlcv[i - 1, 3] - self._ohlcv[i - 1, 0])) + (2 * (self._ohlcv[i - 2, 3] - self._ohlcv[i - 2, 0])) + (self._ohlcv[i - 3, 3] - self._ohlcv[i - 3, 0])) / 6
            rvi_num = np.append(rvi_num, [[num]], axis=0) #RVINumerator = ((Close_i - Open_i) + (2*(Close_(i-1) - Open_(i-1))) + (2*(Close_(i-2) - Open_(i-2))) + (Close_(i-3) - Open_(i-3))) / 6
            den = ((self._ohlcv[i, 1] - self._ohlcv[i, 2]) + (2 * (self._ohlcv[i - 1, 1] - self._ohlcv[i - 1, 2])) + (2 * (self._ohlcv[i - 2, 1] - self._ohlcv[i - 2, 2])) + (self._ohlcv[i - 3, 1] - self._ohlcv[i - 3, 2])) / 6
            rvi_den = np.append(rvi_den, [[den]], axis=0) #RVIDenominator = ((High_i - Low_i) + (2*(High_(i-1) - Low_(i-1))) + (2*(High_(i-2) - Low_(i-2))) + (High_(i-3) - Low_(i-3))) / 6
            self._rvi = np.append(self._rvi, [[np.mean(rvi_num[-self._period:]) / np.mean(rvi_den[-self._period:])]], axis=0) #RVI = SMA(n) of RVINumerator / SMA(n) of RVIDenominator
            sig = (self._rvi[i, 0] + (2 * (self._rvi[i - 1, 0])) + (2 * (self._rvi[i - 2, 0])) + self._rvi[i - 3, 0]) / 6
            self._signal = np.append(self._signal, [[sig]], axis=0) #RVISignal = ((RVI_i) + (2*(RVI_(i-1))) + (2*(RVI(i-2))) + (RVI_(i-3))) / 6

    def goLong(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._rvi[bar] > self._signal[bar]

    def goShort(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._rvi[bar] <= self._signal[bar]

    def getValue(self, bar):
        return self.goLong(bar)

    def firstYieldIndex(self):
        return self._period + 5

