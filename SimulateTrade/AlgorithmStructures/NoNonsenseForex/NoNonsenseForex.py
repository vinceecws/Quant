from ...utils.TradeType import TradeType
from ..Default.AlgorithmStructure import *

class NoNonsenseForex(AlgorithmStructure):

    def __init__(self, indicators):
        super(NoNonsenseForex, self).__init__(indicators)

    def noTrade(self):
        if all(self._indicator_values['volume']):
            if all(self._indicator_values['confirmation']):
                return self._goLong()
            elif not any(self._indicator_values['confirmation']):
                return self._goShort()
            else:
                return self._noAction()
        else:
            return self._noAction()

    def inLong(self):
        if not any(self._indicator_values['exit']):
            return self._exitTrade()
        else: 
            return self._noAction()

    def inShort(self):
        if all(self._indicator_values['exit']):
            return self._exitTrade()
        else: 
            return self._noAction()

            