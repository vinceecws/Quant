from ...utils.TradeType import TradeType
from functools import reduce
import numpy as np

class AlgorithmStructure():

    def __init__(self, indicators):
        self._indicators = indicators #dict of list of Indicators

    @property
    def indicators(self):
        return self._indicators

    def forward(self, indicator_values, trade_state): #returns trade action, or None
        self._indicator_values = indicator_values
        if trade_state == TradeType.long:
            return self.inLong()
        elif trade_state == TradeType.short:
            return self.inShort()
        else:
            return self.noTrade()

    def getFirstBar(self):

        maximum = 0
        for indicators in self._indicators.values():
            new_maximum = reduce(lambda x, y: x if x.firstYieldIndex() > y.firstYieldIndex() else y, indicators)

            if new_maximum.firstYieldIndex() > maximum:
                maximum = new_maximum.firstYieldIndex()

        return maximum

    def _goLong(self):
        return TradeType.long

    def _goShort(self):
        return TradeType.short

    def _exitTrade(self):
        return TradeType.exit_or_no_trade

    def _noAction(self):
        return None

    '''
        The three decision-making methods to be overriden by any algorithm structure:-
        noTrade(), inLong(), inShort()
    '''

    '''
        noTrade:
        Action to be taken when account is not in a trade.

        Parameter(s):
        values: Indicator values
    '''
    def noTrade(self):
        if all(self._indicator_values['confirmation']): #all go long
            return self._goLong()
        elif not any(self._indicator_values['confirmation']): #all go short
            return self._goShort()
        else: 
            return self._noAction()

    '''
        inLong:
        Action to be taken when account is in a long position.

        Parameter(s):
        values: Indicator values
    '''
    def inLong(self):
        if any(i == False for i in self._indicator_values['confirmation']):
            return self._exitTrade()
        else:
            return self._noAction()

    '''
        inShort:
        Action to be taken when account is in a short position.

        Parameter(s):
        values: Indicator values
    '''
    def inShort(self):
        if any(i == True for i in self._indicator_values['confirmation']):
            return self._exitTrade()
        else:
            return self._noAction()

