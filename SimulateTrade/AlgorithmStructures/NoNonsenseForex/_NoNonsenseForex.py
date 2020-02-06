from .AlgorithmStructure import AlgorithmStructure
from functools import reduce
import numpy as np
from .longEntry import longEntry
from .shortEntry import shortEntry

class NoNonsenseForex(AlgorithmStructure):
    
    def __init__(self, ohlcv, risk, confirmation, volume, exit, lowest_denomination, trade_margin=0.05, risk_multiplier=1.5, margin_risk_multiplier=0.02):
        super(NoNonsenseForex, self).__init__(ohlcv, lowest_denomination, trade_margin)
        self._risk = risk #Indicator
        self._confirmation = confirmation #list of Indicators
        self._volume = volume #list of Indicators
        self._exit = exit #list of Indicators
        self._risk_multiplier = risk_multiplier #value for risk multiplier on risk indicator value
        self._margin_risk_multiplier = margin_risk_multiplier #value for risk multiplier on total account margin
        self._firstBar = self.getFirstBar()

    def simulate(self):

        total_bars = 0
        in_trade = False
        trades = 0
        stop_loss_hits = 0
        earnings = np.zeros((self._ohlcv.shape[0], 1), dtype=self._ohlcv.dtype)
        accountValue = np.zeros((max(1, self._firstBar), 1), dtype=self._ohlcv.dtype)

        for bar in np.arange(max(1, self._firstBar), self._lastBar):
            total_bars += 1
            if not in_trade: #Enter new trade
                enter = False
                #Check if all confirmation indicators agree to go long or short
                if all(i.goLong(bar) for i in self._confirmation):
                    enter = True
                    goLong = True
                elif all(i.goShort(bar) for i in self._confirmation):
                    enter = True
                    goLong = False
                else:
                    enter = False

                #Check if all volume indicators are above volume threshold
                if enter and all(i.aboveThreshold(bar) for i in self._volume):
                    enter = True
                else:
                    enter = False

                #Enter trade
                if enter:
                    in_trade = True
                    entrance = self._ohlcv[bar, 3] #Enter trade at closing price
                    risk = self._risk.getValue(bar) * self._risk_multiplier

                    if goLong:
                        stoploss = entrance - risk
                    else:
                        stoploss = entrance + risk

            else: #Manage ongoing trade (either exit, move stop loss to breakeven, or do nothing)
                #Check if any exit indicators predict trend reversal
                #Exit at stop loss
                if goLong:
                    if self._ohlcv[bar, 2] <= stoploss:
                        in_trade = False
                        earnings[bar, 0] = self.pip(-risk)
                        stop_loss_hits += 1
                        trades += 1
                    elif any(i.goShort(bar) for i in self._exit):
                        in_trade = False
                        earnings[bar, 0] = self.pip(self._ohlcv[bar, 3] - entrance)
                        trades += 1

                else:
                    if self._ohlcv[bar, 1] >= stoploss:
                        in_trade = False
                        earnings[bar, 0] = self.pip(-risk)
                        stop_loss_hits += 1
                        trades += 1
                    elif any(i.goLong(bar) for i in self._exit):
                        in_trade = False
                        earnings[bar, 0] = self.pip(entrance - self._ohlcv[bar, 3])
                        trades += 1

                #Move stop loss to breakeven if still in trade and price rises/falls by risk value above entrance price
                if in_trade and goLong and (self._ohlcv[bar, 3] - entrance) >= (entrance - stoploss):
                    stoploss = entrance
                elif in_trade and (not goLong) and (entrance - self._ohlcv[bar, 3]) >= (stoploss - entrance):
                    stoploss = entrance

            #Add profit/loss to account value
            accountValue = np.append(accountValue, [[accountValue[bar - 1, 0] + earnings[bar, 0]]], axis=0)
        
        return total_bars, trades, stop_loss_hits, earnings, accountValue

    def simulateWithRealConditions(self, initial_acc_val, start=None, min_spread=1):
        if initial_acc_val < 500.0:
            raise ValueError('Initial account value must be >= 500.00')

        self._start = start

        if self._start and self._start >= self._firstBar:
            self._start = max(1, start)
        else:
            self._start = max(1, self._firstBar)

        total_bars = 0
        in_trade = False
        trades = 0
        stop_loss_hits = 0
        value_earnings = np.zeros((self._ohlcv.shape[0], 1), dtype=self._ohlcv.dtype)
        half_spread_cost = np.zeros((self._ohlcv.shape[0], 1), dtype=self._ohlcv.dtype)
        margin_cost = np.zeros((self._ohlcv.shape[0], 1), dtype=self._ohlcv.dtype)
        accountValue = np.full((self._start, 1), initial_acc_val, dtype=self._ohlcv.dtype)
        self._net_accountValue = np.full((self._start, 1), initial_acc_val, dtype=self._ohlcv.dtype)
        self._net_earnings = np.zeros((self._ohlcv.shape[0], 1), dtype=self._ohlcv.dtype)

        for bar in np.arange(self._start, self._lastBar):
            if (accountValue[bar - 1, 0] < 0.0):
                print('MARGIN CALL @ bar {}'.format(bar))
                break

            if in_trade: 
                if goLong:
                    change = (self._ohlcv[bar, 3] - self._ohlcv[bar - 1, 3]) * units
                else:
                    change = -(self._ohlcv[bar, 3] - self._ohlcv[bar - 1, 3]) * units
                self._net_accountValue = np.append(self._net_accountValue, [[self._net_accountValue[bar - 1, 0] + change]], axis=0)

            else:
                self._net_accountValue = np.append(self._net_accountValue, [[self._net_accountValue[bar - 1, 0]]], axis=0)

            total_bars += 1
            if not in_trade: #Enter new trade
                enter = False
                #Check if all confirmation indicators agree to go long or short
                if all(i.goLong(bar) for i in self._confirmation):
                    enter = True
                    goLong = True
                elif all(i.goShort(bar) for i in self._confirmation):
                    enter = True
                    goLong = False
                else:
                    enter = False

                #Check if all volume indicators are above volume threshold
                if enter and all(i.aboveThreshold(bar) for i in self._volume):
                    enter = True
                else:
                    enter = False

                #Enter trade
                if enter:
                    in_trade = True
                    rate = self._ohlcv[bar, 3]
                    risk = self._risk.getValue(bar) * self._risk_multiplier

                    #Trading values
                    trade_value = (self._margin_risk_multiplier * accountValue[bar - 1, 0]) / self._trade_margin #The true value of current position
                    units = round(trade_value / rate)

                    if goLong:
                        stop_loss = rate - risk
                        trade = longEntry(rate, units, self._pip_denomination, stop_loss=stop_loss, trade_margin=self._trade_margin)
                        trade_cost, margin_cost[bar, 0], half_spread_cost[bar, 0] = trade.enter(self.getHalfSpread(bar, min_spread))

                    else:
                        stop_loss = rate + risk
                        trade = shortEntry(rate, units, self._pip_denomination, stop_loss=stop_loss, trade_margin=self._trade_margin)
                        trade_cost, margin_cost[bar, 0], half_spread_cost[bar, 0] = trade.enter(self.getHalfSpread(bar, min_spread))

                    #Update account value
                    accountValue = np.append(accountValue, [[(accountValue[bar - 1, 0] - trade_cost)]], axis=0)

                else:
                    accountValue = np.append(accountValue, [[accountValue[bar - 1, 0]]], axis=0)

            else: #Manage ongoing trade (either exit, move stop loss to breakeven, or do nothing)
                #Check if any exit indicators predict trend reversal
                #Exit at stop loss
                new_rate = self._ohlcv[bar, 3]
                exit_at_stop_loss = trade.feedNewRates(self._ohlcv[bar, :], self.getHalfSpread(bar, min_spread)) #Returns None if stop/loss is not triggered
                if exit_at_stop_loss:
                    in_trade = False
                    stop_loss_hits += 1
                    trades += 1

                    #Trading earnings
                    trade_earnings, value_earnings[bar, 0], half_spread_cost[bar, 0], self._net_earnings[bar, 0] = exit_at_stop_loss
                    accountValue = np.append(accountValue, [[(accountValue[bar - 1, 0] + trade_earnings)]], axis=0)

                    if goLong:
                        print('Long Entry @ {} | Stop Loss Exit @ {} | Pips: {} | Units: {} | Cost: {} | Earnings: {}'.format(np.round(rate, 5), np.round(trade.stop_loss, 5), self.pip(trade.stop_loss - rate), units, trade_cost, trade_earnings))
                        change = (trade.stop_loss - self._ohlcv[bar - 1, 3]) * units
                    else:
                        print('Short Entry @ {} | Stop Loss Exit @ {} | Pips: {} | Units: {} | Cost: {} | Earnings: {}'.format(np.round(rate, 5), np.round(trade.stop_loss, 5), self.pip(rate - trade.stop_loss), units, trade_cost, trade_earnings))
                        change = -(trade.stop_loss - self._ohlcv[bar - 1, 3]) * units

                    self._net_accountValue[bar, 0] = self._net_accountValue[bar - 1, 0] + change

                elif (goLong and any(i.goShort(bar) for i in self._exit)) or (not goLong and any(i.goLong(bar) for i in self._exit)):
                    in_trade = False
                    trades += 1

                    #Trading earnings
                    trade_earnings, value_earnings[bar, 0], half_spread_cost[bar, 0], self._net_earnings[bar, 0] = trade.exit(new_rate, self.getHalfSpread(bar, min_spread))
                    accountValue = np.append(accountValue, [[(accountValue[bar - 1, 0] + trade_earnings)]], axis=0)

                    if goLong:
                        print('Long Entry @ {} | Indicator Exit @ {} | Pips: {} | Units: {} | Cost: {} | Earnings: {}'.format(np.round(rate, 5), np.round(self._ohlcv[bar, 3], 5), self.pip(self._ohlcv[bar, 3] - rate), units, trade_cost, trade_earnings))
                    else:
                        print('Short Entry @ {} | Indicator Exit @ {} | Pips: {} | Units: {} | Cost: {} | Earnings: {}'.format(np.round(rate, 5), np.round(self._ohlcv[bar, 3], 5), self.pip(rate - self._ohlcv[bar, 3]), units, trade_cost, trade_earnings))

                else:
                    accountValue = np.append(accountValue, [[accountValue[bar - 1, 0]]], axis=0)

                #Move stop loss to breakeven if still in trade and price rises/falls by risk value above entrance price
                if in_trade and goLong and (self._ohlcv[bar, 3] - rate) >= (rate - trade.stop_loss):
                    trade.stop_loss = rate
                elif in_trade and (not goLong) and (rate - self._ohlcv[bar, 3]) >= (trade.stop_loss - rate):
                    trade.stop_loss = rate

        #Include value of ongoing trades (if any)
        if in_trade:
            value_in_trade, _, _, _ = trade.exit(self._ohlcv[-1, 3], self.getHalfSpread(bar, min_spread))
        else:
            value_in_trade = 0

        return total_bars, trades, stop_loss_hits, value_earnings, half_spread_cost, margin_cost, accountValue, self._net_accountValue, value_in_trade, self._net_earnings

    def getFirstBar(self):
        maximum = self._risk

        maximum = reduce(lambda x, y: x if x.firstYieldIndex() > y.firstYieldIndex() else y, self._confirmation, maximum)
        maximum = reduce(lambda x, y: x if x.firstYieldIndex() > y.firstYieldIndex() else y, self._volume, maximum)
        maximum = reduce(lambda x, y: x if x.firstYieldIndex() > y.firstYieldIndex() else y, self._exit, maximum)

        return maximum.firstYieldIndex()

