from .utils.TradeType import TradeType
from .TradingEnvironments.Default.TradingEnvironment import *
from .RiskStructures.Default.RiskStructure import RiskStructure
import copy

class TradeInterface():

    def __init__(self, trading_environment, algorithm, risk=None, trade_margin=1.0):
        self._trading_environment = trading_environment
        self._algorithm = algorithm

        self._trade_margin = trade_margin

        if risk:
            assert(risk.trade_margin == self._trade_margin)
            self._risk = risk
        else:
            self._risk = RiskStructure(trade_margin=self._trade_margin)

    @property
    def trading_environment(self):
        return self._trading_environment
    
    @property
    def algorithm(self):
        return self._algorithm

    @property
    def trade_margin(self):
        return self._trade_margin
    
    @property
    def risk(self):
        return self._risk

    def forward(self, close):
        if self._trading_environment.isEnd():
            return

        values = self._getIndicatorValues(self._trading_environment.bar)

        if isinstance(self._trading_environment.trade, LongEntry): 
            trade_state = TradeType.long
        elif isinstance(self._trading_environment.trade, ShortEntry):
            trade_state = TradeType.short
        else:
            trade_state = TradeType.exit_or_no_trade

        action = self._algorithm.forward(values, trade_state)

        if action:
            if action == TradeType.long: 
                self._trading_environment.enterLong(self._risk.getUnits(self._trading_environment.bar, self._trading_environment.getCurrentPrice(), self._trading_environment.getCurrentAccountValue()), \
                    self._risk.getLongStopLoss(self._trading_environment.bar, self._trading_environment.getCurrentPrice()))
            elif action == TradeType.short:
                self._trading_environment.enterShort(self._risk.getUnits(self._trading_environment.bar, self._trading_environment.getCurrentPrice(), self._trading_environment.getCurrentAccountValue()), \
                    self._risk.getShortStopLoss(self._trading_environment.bar, self._trading_environment.getCurrentPrice()))
            elif action == TradeType.exit_or_no_trade:
                self._trading_environment.exit()
            else:
                pass

    def _getIndicatorValues(self, bar):
        indicator_values = copy.deepcopy(self._algorithm.indicators)
        for indicator_type, indicators in self._algorithm.indicators.items():
            if type(indicators) is list:
                values = []
                for indicator in indicators:
                    values.append(indicator.getValue(bar))
                indicator_values.update({indicator_type:values})
            else:
                indicator_values.update({indicator_type:indicator.getValue(bar)})

        return indicator_values

    def _goingLong(self):
        if isinstance(self._trading_environment.trade, LongEntry):
            return True
        else:
            return False 

    def _goingShort(self):
        if isinstance(self._trading_environment.trade, ShortEntry):
            return True
        else:
            return False

    def printResults(self):

        self._trading_environment.printResults()

    def getResults(self):

        return self._trading_environment.getResults()

    def printMetrics(self):

        return self._trading_environment.printMetrics()

        