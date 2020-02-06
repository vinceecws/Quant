import numpy as np

class Indicators():
    _name = 'Indicator'

    def __init__(self, ohlcv, period):
        self._ohlcv = np.array(ohlcv)
        self._period = period
        self.checkPeriod()

    @property
    def ohlcv(self):
        return self._ohlcv
    
    @property
    def period(self):
        return self._period

    @property
    def name(self):
        return self._name
    
    def __str__(self):
        return self._name + ' ' + str(self._period)

    def checkPeriod(self):
        if (self._ohlcv.shape[0] < self._period):
            raise ValueError('Data given has period of {}, while desired period is {}'.format(self._ohlcv.shape[0], self._period))

    def getValue(self, bar):
        return 0

    def firstYieldIndex(self):
        return 0







