import numpy as np
from .Indicators import Indicators

class AverageTrueRange(Indicators):
    _name = 'AverageTrueRange'

    def __init__(self, ohlcv, period=14):
        super(AverageTrueRange, self).__init__(ohlcv, period)
        self.generateData()

    def __str__(self):
        return self._name + ' ' + str(self._period)

    @property
    def average_true_range(self):
        return self._average_true_range

    @property
    def true_range(self):
        return self._true_range

    '''
    Average True Range:
    Indicates the volatility of the market

    Output(s):
    Average True Range: (previous Average True Range * (period - 1) + current True Range) / period
        *True Range: max(current high - current low, |current high - previous close|, |current low - previous close|)
    '''

    def generateData(self):

        self._true_range = np.empty((0, 1), dtype=self._ohlcv.dtype) #High-Low for first n (period) days
        for i in np.arange(0, self._period):
            high_low = self._ohlcv[i, 1] - self._ohlcv[i, 2]
            self._true_range = np.append(self._true_range, [[high_low]], axis=0)

        self._average_true_range = np.zeros((self._period - 1, 1), dtype=self._ohlcv.dtype)
        self._average_true_range = np.append(self._average_true_range, [np.mean(self._true_range, axis=0)], axis=0)

        for i in np.arange(self._period, self._ohlcv.shape[0]):
            high_low = self._ohlcv[i, 1] - self._ohlcv[i, 2] #current high - current low
            high_close = abs(self._ohlcv[i, 1] - self._ohlcv[i - 1, 3]) #|current high - previous close|
            low_close = abs(self._ohlcv[i, 2] - self._ohlcv[i - 1, 3]) #|current low - previous close|
            true = np.max([high_low, high_close, low_close])
            self._true_range = np.append(self._true_range, [[true]], axis=0)
            current = ((self._average_true_range[i - 1, 0] * (self._period - 1)) + true) / self._period
            self._average_true_range = np.append(self._average_true_range, [[current]], axis=0)

    def getValue(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._average_true_range[bar, 0]

    def firstYieldIndex(self):
        return self._period - 1
