import argparse
import numpy as np
from matplotlib import pyplot as plt
from SimulateTrade.NoNonsenseForex import NoNonsenseForex
from utils.ohlcv import OHLCV
from utils.Indicators.AverageTrueRange import AverageTrueRange
from utils.Indicators.AroonUpAndDown import AroonUpAndDown
from utils.Indicators.AverageDirectionalIndex import AverageDirectionalIndex
from utils.Indicators.MarketFacilitationIndex import MarketFacilitationIndex
from utils.Indicators.RelativeVigorIndex import RelativeVigorIndex
from utils.Indicators.PercentageVolumeOscillator import PercentageVolumeOscillator

def main(args):
    chart_data_dir = args.chart_data_dir
    init_margin = args.init_margin

    print('============================================================ ')
    print('Preparing chart and indicator data...')
    data = OHLCV(chart_data_dir, window=30).data
    print('Chart ready.')
    risk = AverageTrueRange(data)
    confirmation = [AverageDirectionalIndex(data), AroonUpAndDown(data)]
    volume = [MarketFacilitationIndex(data)]
    exit = [RelativeVigorIndex(data, 5)]
    print('Indicator ready.')
    print('============================================================ ')
    print('Starting algorithm engine...')
    algorithm = NoNonsenseForex(data, risk, confirmation, volume, exit, 0.01, risk_multiplier=1.0, margin_risk_multiplier=0.90)
    print('Algorithm engine ready.')
    print('============================================================ ')
    print('Running simulation...')
    total_bars, trades, stop_loss_hits, value_earnings, half_spread_cost, margin_cost, accountValue, net_accountValue, value_in_trade, net_earnings = algorithm.simulateWithRealConditions(init_margin, start=2029, min_spread=1)
    print('Simulation complete.')

    print('============================================================ ')
    print('STRUCTURE:')
    print('Risk Indicator: {}'.format(str(risk)))
    print('Confirmation Indicator(s): {}'.format(', '.join(str(i) for i in confirmation)))
    print('Volume Indicator(s): {}'.format(', '.join(str(i) for i in volume)))
    print('Exit Indicator(s): {}'.format(', '.join(str(i) for i in exit)))
    print('============================================================ ')
    print('RESULTS:')
    print('No. of bars: {}'.format(total_bars))
    print('No. of trades: {}'.format(trades))
    print('Stop/Loss hits: {}'.format(stop_loss_hits))
    print('Realized P/L: {}'.format(accountValue[-1, 0] - init_margin))
    if (value_in_trade != 0):
        print('Unrealized P/L: {}'.format(value_in_trade))
    else:
        print('Unrealized P/L: {}'.format(0))
    print('Account value: {}'.format(net_accountValue[-1, 0]))
    print('Actual account value (spread cost adjusted): {}'.format(accountValue[-1, 0]))
    print('Return: {:.2f}%'.format(((accountValue[-1, 0] + value_in_trade - init_margin) / init_margin) * 100))
    print('Total spread cost: {}'.format(np.sum(half_spread_cost)))
    print('============================================================ ')
    print('METRICS:')
    print('Profit Factor: {}'.format(algorithm.profitFactor()))
    print('Compounded Annual Return (CAR): {:.2f}%'.format(algorithm.CAR() * 100))
    print('Maximum Drawdown (MDD): {:.2f}%'.format(algorithm.MDD() * 100))
    print('Measurement of Returns Adjusted for Risk (MAR): {}'.format(algorithm.MAR()))

    fig = plt.figure(figsize=(5,10))
    plt.subplot(121)
    plt.plot(net_earnings)
    plt.xlabel('Bar')
    plt.ylabel('Earnings')

    plt.subplot(122)
    plt.plot(net_accountValue)
    plt.xlabel('Bar')
    plt.ylabel('Account Value')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart_data_dir", type=str, default="/Users/vincentchooi/desktop/quant/chart_data/fx/USDCHF_Candlestick_1_D_BID_06.06.2009-01.06.2019.csv", required=False, help="File: Chart data in CSV")
    parser.add_argument("--init_margin", type=int, default=10000, required=False, help="Initial margin value")
    args = parser.parse_args()

    main(args)

