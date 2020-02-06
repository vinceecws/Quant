import numpy as np
from .Indicators import Indicators
from .AverageTrueRange import AverageTrueRange

class AverageDirectionalIndex(Indicators):
    _name = 'AverageDirectionalIndex'

    def __init__(self, ohlcv, period=14):
        super(AverageDirectionalIndex, self).__init__(ohlcv, period)
        self._atr = AverageTrueRange(ohlcv, period=period)
        self.generateData()

    def __str__(self):
        return self._name + ' ' + str(self._period)

    @property
    def adx(self):
        return self._adx
    
    @property
    def plus_di_per(self):
        return self._plus_di_per
    
    @property
    def minus_di_per(self):
        return self._minus_di_per

    '''
    Average Directional Index:
    Determines the strength of a trend.

    Output(s):
    Average Directional Index: (previous ADX * (period - 1) + current DX) / period
        *Directional Index: (abs(+DI - -DI) / abs(+DI + -DI)) * 100

    +Directional Indicator: (+DMPer / TRPer) * 100
        *+Directional Movement: previous +DMPer - (previous +DMPer / period) + current +DM

    -Directional Indicator: (-DMPer / TRPer) * 100
        *-Directional Movement: previous -DMPer - (previous -DMPer / period) + current -DM
        *True Range (Smoothed): previous TRPer - (previous TRPer / period) + current TR
    '''

    def generateData(self):
        tr = self._atr._true_range

        plus_dm = np.zeros((1, 1), dtype=self._ohlcv.dtype)
        minus_dm = np.zeros((1, 1), dtype=self._ohlcv.dtype)

        for i in np.arange(1, self._ohlcv.shape[0]):
            up_move = self._ohlcv[i, 1] - self._ohlcv[i - 1, 1] #current high - previous high
            down_move = self._ohlcv[i - 1, 2] - self._ohlcv[i, 2]#previous low - current low

            if up_move > down_move and up_move > 0.0:
                plus_dm = np.append(plus_dm, [[up_move]], axis=0)
                minus_dm = np.append(minus_dm, [[0.0]], axis=0)
            elif down_move > up_move and down_move > 0.0:
                plus_dm = np.append(plus_dm, [[0.0]], axis=0)
                minus_dm = np.append(minus_dm, [[down_move]], axis=0)
            else:
                plus_dm = np.append(plus_dm, [[0.0]], axis=0)
                minus_dm = np.append(minus_dm, [[0.0]], axis=0)

        #Initialize initial periods
        tr_per = np.zeros((self._period, 1), dtype=self._ohlcv.dtype)
        plus_dm_per = np.zeros((self._period, 1), dtype=self._ohlcv.dtype)
        minus_dm_per = np.zeros((self._period, 1), dtype=self._ohlcv.dtype)
        self._plus_di_per = np.zeros((self._period, 1), dtype=self._ohlcv.dtype)
        self._minus_di_per = np.zeros((self._period, 1), dtype=self._ohlcv.dtype)
        dx = np.zeros((self._period, 1), dtype=self._ohlcv.dtype)
        self._adx = np.zeros((2 * self._period - 1, 1), dtype=self._ohlcv.dtype)

        #First period calculations
        tr_per = np.append(tr_per, [[np.sum(tr)]], axis=0) #TRPer = sum(first n (period) TR)
        plus_dm_per = np.append(plus_dm_per, [[np.sum(plus_dm)]], axis=0) #+DMPer = sum(first n (period) +DM)
        minus_dm_per = np.append(minus_dm_per, [[np.sum(minus_dm)]], axis=0) #-DMPer = sum(first n (period) -DM)
        self._plus_di_per = np.append(self._plus_di_per, [[((plus_dm_per[-1, 0] / tr_per[-1, 0]) * 100)]], axis=0) #+DIPer = (+DMPer / TRPer) * 100
        self._minus_di_per = np.append(self._minus_di_per, [[((minus_dm_per[-1, 0] / tr_per[-1, 0]) * 100)]], axis=0) #-DIPer = (-DMPer / TRPer) * 100
        dx = np.append(dx, [[((abs(self._plus_di_per[-1, 0] - self._minus_di_per[-1, 0]) / abs(self._plus_di_per[-1, 0] + self._minus_di_per[-1, 0])) * 100)]], axis=0) #DX = (abs(+DI - -DI) / abs(+DI + -DI)) * 100

        for i in np.arange(self._period, 2 * self._period - 1):
            tr_per = np.append(tr_per, [[(tr_per[i - 1, 0] - (tr_per[i - 1, 0] / self._period) + tr[i, 0])]], axis=0) #TRPer = previous TRPer - (previous TRPer / period) + current TR
            plus_dm_per = np.append(plus_dm_per, [[(plus_dm_per[i - 1, 0] - (plus_dm_per[i - 1, 0] / self._period) + plus_dm[i, 0])]], axis=0) #+DMPer = previous +DMPer - (previous +DMPer / period) + current +DM
            minus_dm_per = np.append(minus_dm_per, [[(minus_dm_per[i - 1, 0] - (minus_dm_per[i - 1, 0] / self._period) + minus_dm[i, 0])]], axis=0) #-DMPer = previous -DMPer - (previous -DMPer / period) + current -DM
            self._plus_di_per = np.append(self._plus_di_per, [[((plus_dm_per[i, 0] / tr_per[i, 0]) * 100)]], axis=0) #+DIPer = (+DMPer / TRPer) * 100
            self._minus_di_per = np.append(self._minus_di_per, [[((minus_dm_per[i, 0] / tr_per[i, 0]) * 100)]], axis=0) #-DIPer = (-DMPer / TRPer) * 100
            dx = np.append(dx, [[((abs(self._plus_di_per[i, 0] - self._minus_di_per[i, 0]) / abs(self._plus_di_per[i, 0] + self._minus_di_per[-1, 0])) * 100)]], axis=0) #DX = (abs(+DI - -DI) / abs(+DI + -DI)) * 100

        self._adx = np.append(self._adx, [np.mean(dx, axis=0)], axis=0) #ADXPer = mean(first n (period) DX)

        for i in np.arange(2 * self._period, self._ohlcv.shape[0]):
            tr_per = np.append(tr_per, [[(tr_per[i - 1, 0] - (tr_per[i - 1, 0] / self._period) + tr[i, 0])]], axis=0) #TRPer = previous TRPer - (previous TRPer / period) + current TR
            plus_dm_per = np.append(plus_dm_per, [[(plus_dm_per[i - 1, 0] - (plus_dm_per[i - 1, 0] / self._period) + plus_dm[i, 0])]], axis=0) #+DMPer = previous +DMPer - (previous +DMPer / period) + current +DM
            minus_dm_per = np.append(minus_dm_per, [[(minus_dm_per[i - 1, 0] - (minus_dm_per[i - 1, 0] / self._period) + minus_dm[i, 0])]], axis=0) #-DMPer = previous -DMPer - (previous -DMPer / period) + current -DM
            self._plus_di_per = np.append(self._plus_di_per, [[((plus_dm_per[i, 0] / tr_per[i, 0]) * 100)]], axis=0) #+DIPer = (+DMPer / TRPer) * 100
            self._minus_di_per = np.append(self._minus_di_per, [[((minus_dm_per[i, 0] / tr_per[i, 0]) * 100)]], axis=0) #-DIPer = (-DMPer / TRPer) * 100
            dx = np.append(dx, [[((abs(self._plus_di_per[i, 0] - self._minus_di_per[i, 0]) / abs(self._plus_di_per[i, 0] + self._minus_di_per[-1, 0])) * 100)]], axis=0) #DX = (abs(+DI - -DI) / abs(+DI + -DI)) * 100
            self._adx = np.append(self._adx, [[((self._adx[i - 1, 0] * (self._period - 1)) + dx[i, 0]) / self._period]], axis=0) #ADXPer = (previous ADXPer * (period - 1) + current DX) / period

    def goLong(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._plus_di_per[bar] > self._minus_di_per[bar]

    def goShort(self, bar):
        if bar < self.firstYieldIndex():
            raise ValueError('bar index must be >= first yield index')
        return self._plus_di_per[bar] <= self._minus_di_per[bar]

    def getValue(self, bar):
        return self.goLong(bar)

    def firstYieldIndex(self):
        return 2 * self._period - 1
