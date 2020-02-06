import numpy as np
from .Indicators import Indicators

class ExponentialMovingAverage(Indicators):
    _name = 'ExponentialMovingAverage'

    def __init__(self, ohlcv, period=14, col=3):
        super(ExponentialMovingAverage, self).__init__(ohlcv, period)
        self._col = col
        self.generateData()

    def __str__(self):
        return self._name + ' ' + str(self._period)

    @property
    def ema(self):
        return self._ema

    '''
    The exponential moving average (EMA) is a type of moving average (MA) that places a greater weight and significance on the most recent data points. 
    Exponential moving average is also referred to as the exponentially weighted moving average. 

    Output(s):
    Exponential Moving Average: (Close - previous EMA) x multiplier + previous EMA
        **First EMA = SMA(period)
    '''

    def generateData(self):
        self._ema = np.zeros((self._period - 1, 1), dtype=self._ohlcv.dtype)
        
        #Initial ema = sma(period)
        sma = np.sum(self._ohlcv[0:self._period - 1, self._col]) / self._period
        self._ema = np.append(self._ema, [[sma]], axis=0)

        multiplier = 2 / (self._period + 1) #Smoothing constant

        for i in np.arange(self._period, self._ohlcv.shape[0]):
            val = (self._ohlcv[i, self._col] - self._ema[i - 1, 0]) * multiplier + self._ema[i - 1, 0] #EMA: (Close - previous EMA) x multiplier + previous EMA
            self._ema = np.append(self._ema, [[val]], axis=0) 

    def firstYieldIndex(self):
        return self._period - 1

    def getValue(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._ema[bar, 0]
        

