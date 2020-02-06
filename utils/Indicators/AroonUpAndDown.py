import numpy as np
from .Indicators import Indicators

class AroonUpAndDown(Indicators):
    _name = 'AroonUpAndDown'

    def __init__(self, ohlcv, period=14):
        super(AroonUpAndDown, self).__init__(ohlcv, period)
        self.generateData()

    def __str__(self):
        return self._name + ' ' + str(self._period)

    @property
    def aroon_up(self):
        return self._aroon_up

    @property
    def aroon_down(self):
        return self._aroon_down
    
    @property
    def aroon_osc(self):
        return self._aroon_osc

    '''
    Aroon Oscillator:
    Determines the strength of the current trend, and the likelihood that it will continue

    Output(s):
    Aroon Up: ((period - (periods since highest high)) / period) * 100

    Aroon Down: ((period - (periods since lowest low)) / period) * 100

    Aroon Oscillator: Aroon Up - Aroon Down
    '''

    def generateData(self):

        self._aroon_up = np.zeros((self._period - 1, 1), dtype=self._ohlcv.dtype)
        self._aroon_down = np.zeros((self._period - 1, 1), dtype=self._ohlcv.dtype)
        self._aroon_osc = np.zeros((self._period - 1, 1), dtype=self._ohlcv.dtype)

        for i in np.arange(self._period - 1, self._ohlcv.shape[0]): #stop using yield_length
            high = self._ohlcv[i - (self.period - 1): i, 1]
            highest = np.argmax(high)
            low = self._ohlcv[i - (self.period - 1): i, 2]
            lowest = np.argmin(low)
            up = ((self._period - highest) / self._period) * 100
            down = ((self._period - lowest) / self._period) * 100
            self._aroon_up = np.append(self._aroon_up, [[up]], axis=0)
            self._aroon_down = np.append(self._aroon_down, [[down]], axis=0)
            self._aroon_osc = np.append(self._aroon_osc, [[up - down]], axis=0)

    def goLong(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._aroon_up[bar] > self._aroon_down[bar]

    def goShort(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._aroon_up[bar] <= self._aroon_down[bar]

    def getValue(self, bar):
        return self.goLong(bar)

    def firstYieldIndex(self):
        return self._period - 1
