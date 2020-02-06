class RiskStructure():

    def __init__(self, risk_indicators=None, trade_margin=1.0):
        self._risk_indicators = risk_indicators #list of Indicators

        if trade_margin > 1.0 or trade_margin < 0.0:
            raise ValueError('Trade margin % must be between 0% - 100%')

        self._trade_margin = trade_margin

    @property
    def risk_indicators(self):
        return self._risk_indicators

    @property
    def trade_margin(self):
        return self._trade_margin

    '''
        The three risk-calculation methods to be overriden by any risk structure:-
        getLongStopLoss(), getShortStopLoss(), getUnits()
    '''

    '''
        getLongStopLoss:
        Returns the optimal stop loss value for a given long trade

        Parameter(s):
        current_price: Current price of security

        Return(s):
        stop_loss: Price value for stop loss to be set at
    '''
    def getLongStopLoss(self, bar, current_price):

        return max(0.0, current_price - 0.5)

    '''
        getShortStopLoss:
        Returns the optimal stop loss value for a given short trade

        Parameter(s):
        current_price: Current price of security

        Return(s):
        stop_loss: Price value for stop loss to be set at
    '''
    def getShortStopLoss(self, bar, current_price):

        return current_price + 0.5

    '''
        getUnits:
        Returns the optimal no. of units to be traded for a given security, based on current account value

        Parameter(s):
        current_price: Current price of security
        current_account_value: Current value of trading account

        Return(s):
        units: Optimal units to be traded

    '''

    def getUnits(self, bar, current_price, current_account_value, risk_percentage=0.02):

        if risk_percentage > 1.0 or risk_percentage < 0.0:
            raise ValueError('Risk % must be between 0% - 100%')

        risk = current_account_value * risk_percentage
        units = risk / current_price
        units = units / self._trade_margin

        return units
