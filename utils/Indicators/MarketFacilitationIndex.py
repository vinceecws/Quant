import numpy as np
from .Indicators import Indicators

class MarketFacilitationIndex(Indicators):
    _name = 'MarketFacilitationIndex'

    def __init__(self, ohlcv, period=1):
        super(MarketFacilitationIndex, self).__init__(ohlcv, period)
        if period != 1:
            raise ValueError('Period must be 1')
        self.generateData()

    def __str__(self):
        return self._name + ' ' + str(self._period)

    @property
    def mfi(self):
        return self._mfi

    @property
    def color(self):
        return self._color
    

    '''
    Market Facilitation Index:
    Market Facilitation Index Technical Indicator (BW MFI) is the indicator which shows the change of price for one tick. 
    Absolute values of the indicator do not mean anything as they are, only indicator changes have sense.
    The changes are compared with the previous bar, represented in four colors:-
        [0] - MFI decreases, volume decreases
        [1] - MFI decreases, volume increases
        [2] - MFI increases, volume decreases
        [3] - MFI increases, volume increases

    Output(s):
    Market Facilitation Index: (High - Low) / Volume
    Color: 0, 1, 2 or 3
    '''

    def generateData(self):

        #Initialize initial periods
        self._mfi = np.empty((0, 1), dtype=self._ohlcv.dtype)
        self._color = np.zeros((1, 1), dtype=np.int32)

        #First period calculation
        val = (self._ohlcv[0, 1] - self._ohlcv[0, 2]) / self._ohlcv[0, 4] #MFI = (High - Low) / Volume
        self._mfi = np.append(self._mfi, [[val]], axis=0)

        for i in range(1, self._ohlcv.shape[0]):
            val = (self._ohlcv[i, 1] - self._ohlcv[i, 2]) / self._ohlcv[i, 4] #MFI = (High - Low) / Volume
            self._mfi = np.append(self._mfi, [[val]], axis=0)

            if self._mfi[i, 0] < self._mfi[i - 1, 0]: 

                if self._ohlcv[i, 4] < self._ohlcv[i - 1, 4]: #MFI decreases, volume decreases
                    self._color = np.append(self._color, [[0]], axis=0)

                else: #MFI decreases, volume increases
                    self._color = np.append(self._color, [[1]], axis=0)

            else:

                if self._ohlcv[i, 4] < self._ohlcv[i - 1, 4]: #MFI increases, volume decreases
                    self._color = np.append(self._color, [[2]], axis=0)

                else: #MFI increases, volume increases
                    self._color = np.append(self._color, [[3]], axis=0)

    def firstYieldIndex(self):
        return 1

    def aboveThreshold(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._color[bar] == 3 #If color is green

    def getValue(self, bar):
        return self.aboveThreshold(bar)








