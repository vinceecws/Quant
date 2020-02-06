from ..Default.RiskStructure import RiskStructure
import sys
sys.path.append('....')
from utils.Indicators.AverageTrueRange import AverageTrueRange

class ATRisk(RiskStructure):

    def __init__(self, risk_indicators=None, trade_margin=1.0, risk_multiplier=1.5):
        super(ATRisk, self).__init__(risk_indicators, trade_margin)

        if type(self._risk_indicators[0]) is not AverageTrueRange:
            raise ValueError('Structure requires the Average True Range indicator')

        if risk_multiplier <= 0.0:
            raise ValueError('Risk multiplier must be > 0')

        self._risk_multiplier = risk_multiplier

    @property
    def risk_multiplier(self):
        return self._risk_multiplier

    def getLongStopLoss(self, bar, current_price):
        risk = self._risk_indicators[0].getValue(bar)

        return max(0.0, current_price - (self._risk_multiplier * risk))

    def getShortStopLoss(self, bar, current_price):
        risk = self._risk_indicators[0].getValue(bar)

        return current_price + (self._risk_multiplier * risk)