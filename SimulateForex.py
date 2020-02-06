import argparse
import numpy as np
from matplotlib import pyplot as plt
from SimulateTrade.TradingEnvironments.Forex.ForexTradingEnvironment import ForexTradingEnvironment
from SimulateTrade.AlgorithmStructures.Default.AlgorithmStructure import AlgorithmStructure
from SimulateTrade.AlgorithmStructures.NoNonsenseForex.NoNonsenseForex import NoNonsenseForex
from SimulateTrade.RiskStructures.ATRisk.ATRisk import ATRisk
from SimulateTrade.TradeInterface import TradeInterface
from SimulateTrade.TradeEngine import TradeEngine
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
    risk = [AverageTrueRange(data)]
    confirmation = [AverageDirectionalIndex(data), AroonUpAndDown(data)]
    volume = [MarketFacilitationIndex(data)]
    exit = [RelativeVigorIndex(data)]
    indicators = {'risk': risk, 'confirmation': confirmation, 'volume': volume, 'exit': exit}
    print('Indicators ready.')
    print('Initializing algorithm and trading environment...')
    algorithm_structure = NoNonsenseForex(indicators)
    print('Algorithm ready.')
    risk_structure = ATRisk([AverageTrueRange(data)], trade_margin=0.05, risk_multiplier=1.0)
    print('Risk ready.')
    trading_environment = ForexTradingEnvironment(data, algorithm_structure.getFirstBar(), 0.01, min_spread=1, initial_account_value=init_margin, trade_margin=0.05)
    print('Trading environment ready.')
    print('Linking algorithm, risk and environment...')
    interface = TradeInterface(trading_environment, algorithm_structure, risk_structure, trade_margin=0.05)
    print('Starting trade engine...')
    engine = TradeEngine(interface)
    print('Simulation start.')
    engine.run()
    print('Simulation complete.')
    print('============================================================ ')
    interface.printResults()
    print('============================================================ ')
    interface.printMetrics()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart_data_dir", type=str, default="/Users/vincentchooi/desktop/quant/chart_data/fx/GBPUSD_Candlestick_1_D_BID_06.06.2009-01.06.2019.csv", required=False, help="File: Chart data in CSV")
    parser.add_argument("--init_margin", type=int, default=10000, required=False, help="Initial margin value")
    args = parser.parse_args()

    main(args)