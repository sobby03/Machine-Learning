# https://github.com/backtrader/backtrader/tree/master/backtrader/indicators

# All the necessary libraries
import talib as ta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy
import warnings
# from datetime import datetime
import logbook
from logbook import Logger

log = Logger('Algorithm')
import pytz
from zipline.algorithm import TradingAlgorithm
# from zipline.utils.factory import load_from_yahoo
from zipline.api import order, symbol, get_order, record
import pylab as pl
import time

warnings.filterwarnings("ignore")

# from datetime import datetime, timedelta


class RSI_Strategy:
    def initialize(context):
        context.test = 1
        context.stock = symbol(sec)
        context.RSI_upper = 65
        context.RSI_lower = 35
        # Flag for when the RSI is in the overbought region
        context.RSI_OB = False
        # Flag for when the RSI is in the oversold region
        context.RSI_OS = False
        context.buy = False
        context.sell = False


    def handle_data(self, context, data):
        try:
            # trailing_window = data.history(context.stock, 'price', 35, '1d')
            RSI = self.rsiFunc(data['close'])
        except:
            return

        crossover_flag = False
        RSI = ta.RSI(trailing_window.values, 14)

        # Setting flags
        if (RSI[-1] > context.RSI_upper):
            context.RSI_OB = True

        elif (RSI[-1] < context.RSI_lower):
            context.RSI_OS = True

        if (RSI[-1] < context.RSI_upper and context.RSI_OB):
            context.RSI_OB = False
            crossover_flag = True

        elif (RSI[-1] > context.RSI_lower and context.RSI_OS):
            context.RSI_OS = False
            crossover_flag = True

        # Trading Logic
        if (crossover_flag and RSI[-1] < 50 and not context.buy):
            context.order_target(context.stock, 100)
            context.buy = True

        if (crossover_flag and RSI[-1] > 50 and not context.sell):
            context.order_target(context.stock, -100)
            context.sell = True

        if (context.buy and RSI[-1] >= 50):
            context.order_target(context.stock, 0)
            context.buy = False
            context.sell = False

        if (context.sell and RSI[-1] <= 50):
            context.order_target(context.stock, 0)
            context.buy = False
            context.sell = False

            # Recording Results
        record(security=data[symbol(sec)].price,
               RSI=RSI[-1],
               buy=context.buy,
               sell=context.sell)

    def analyze(self, context, perf):
        fig = plt.figure()

        # Set up a plot of the portfolio value
        ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=3, colspan=1)
        perf.portfolio_value.plot(ax=ax1)
        ax1.set_ylabel('Portfolio Value ($)')

        # Set up a plot of the security value
        ax2 = plt.subplot2grid((8, 1), (3, 0), rowspan=3, colspan=1, sharex=ax1)
        data.plot(ax=ax2)
        ax2.set_ylabel(sec + ' Value ($)')

        # Find transaction points
        perf_trans = perf.ix[[t != [] for t in perf.transactions]]
        buys = perf_trans.ix[[t[0]['amount'] > 0 for t in perf_trans.transactions]]
        sells = perf_trans.ix[[t[0]['amount'] < 0 for t in perf_trans.transactions]]

        # Plot the colored box of the final transaction if the time period ends with an open position
        if (len(buys) != len(sells)):
            if (len(buys) > len(sells)):
                upper_lim = len(sells)
                last_point = mdates.date2num(buys.index[upper_lim])
                col = 'g'
            elif (len(sells) < len(buys)):
                upper_lim = len(buys)
                last_point = mdates.date2num(sells.index[upper_lim])
                col = 'r'
            end_d = mdates.date2num(end)
            width = mdates.date2num(end) - last_point
            rect = Rectangle((last_point, 60), width, 40, color=col, alpha=0.3)
            ax2.add_patch(rect)
        else:
            upper_lim = len(buys)

        short_patch = mpatches.Patch(color='r', alpha=0.3, label='Short Holdings')
        long_patch = mpatches.Patch(color='g', alpha=0.3, label='Long Holdings')
        plt.legend(handles=[long_patch, short_patch])
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='8')

        # Plot the colored box of all other transactions
        for i in range(0, upper_lim):
            buy_d = mdates.date2num(buys.index[i])
            sell_d = mdates.date2num(sells.index[i])
            if (buy_d < sell_d):
                col = 'g'
                width = sell_d - buy_d
                rect = Rectangle((buy_d, 60), width, 40, color=col, alpha=0.3)
            else:
                col = 'r'
                width = buy_d - sell_d
                rect = Rectangle((sell_d, 60), width, 40, color=col, alpha=0.3)
            ax2.add_patch(rect)

        # Plot the RSI with proper lines and shading
        ax3 = plt.subplot2grid((8, 1), (6, 0), rowspan=2, colspan=1, sharex=ax1)
        ax3.plot(perf['RSI'])
        ax3.fill_between(perf.index, perf['RSI'], context.RSI_upper,
                         where=perf['RSI'] >= context.RSI_upper, alpha=0.5, color='r')
        ax3.fill_between(perf.index, perf['RSI'], context.RSI_lower,
                         where=perf['RSI'] <= context.RSI_lower, alpha=0.5, color='g')
        ax3.plot((date2num(perf.index[0]), date2num(perf.index[-1])),
                 (context.RSI_upper, context.RSI_upper), color='r', alpha=0.5)
        ax3.plot((date2num(perf.index[0]), date2num(perf.index[-1])),
                 (context.RSI_lower, context.RSI_lower), color='g', alpha=0.5)
        ax3.grid(True)
        ax3.set_ylabel('RSI')

        OB_patch = mpatches.Patch(color='r', alpha=0.5, label='Overbought')
        OS_patch = mpatches.Patch(color='g', alpha=0.5, label='Oversold')
        plt.legend(loc='upper left', handles=[OB_patch, OS_patch])
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='8')

        plt.tight_layout()
        plt.show()

def convertToMilliSec(intraTime):
    # time.mktime(intraTime.timetuple()) * 1000
    # return time.mktime(intraTime) * 1000
    return int(intraTime) * 1000

def getIntradayCSV(symbol):
    fmt = "%Y-%m-%d %H:%M:%S"
    data = pd.read_csv('data/' + symbol + '.csv')
    data.index = pd.to_datetime(data.date, format=fmt)
    todays_data = data.loc['2018-06-05']
    # todays_data['time'] = todays_data.index.strftime('%s')
    # todays_data.reset_index(drop=True, inplace=True)
    # todays_data['time'].apply(convertToMilliSec)
    # todays_data['time'] = todays_data.time.astype(float)
    # print("Time :: ", time.asctime(time.localtime(time.time())))
    # return todays_data.iloc[0:80]
    return todays_data

# Choosing a security and a time horizon
logbook.StderrHandler().push_application()
# start = datetime(2012, 9, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2016, 1, 1, 0, 0, 0, 0, pytz.utc)
sec = 'TATASTEEL'
data = getIntradayCSV(sec)

data = data.dropna()
algo = TradingAlgorithm(initialize=RSI_Strategy.initialize,
                        handle_data=RSI_Strategy.handle_data,
                        analyze=RSI_Strategy.analyze)
results = algo.run(data)