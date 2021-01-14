import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from datetime import date
import yfinance as yf
import datetime as dt


def computeBollinger(close, period, std = 2):
    df = pd.DataFrame()
    MA = 'MA' + str(period)
    STD = str(period) + 'STD'
    df[MA] = close.rolling(window=period).mean()
    df[STD] = close.rolling(window=period).std()
    df['upper'] = df[MA] + (df[STD] * std)
    df['lower'] = df[MA] - (df[STD] * std)
    return df[MA], df['upper'], df['lower']

def calculate_limits(ticker, tradeperiod=10, frequency='15m'): # valid fetch periods “1m”, “2m”, “5m”, “15m”, “30m”, “60m”, “90m”, “1h”
    data = pd.DataFrame()
    today = date.today()
    start_date = today - dt.timedelta(days=7)
    
    asset = yf.Ticker(ticker)
    ohclv = asset.history(start=start_date, interval=frequency)

    data[ticker] = ohclv['Close']#wb.DataReader(ticker, data_source='yahoo', start=start_date, end=today)['Close']
    
    
    
    #data = data['Close']
    
    data_row = data.shape
    log_returns = np.log(1 + data.pct_change())

    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    #print(' DEBUG: natual log of mean daily returns: ' + str((float(u))) + '%')
    #print('DEBUG: variance: ' + str(float(var)) + '%')

    stdev = log_returns.std()
    days = tradeperiod
    trials = 10000
    Z = norm.ppf(np.random.rand(days, trials))  # days, trials
    daily_returns = np.exp(drift.values + stdev.values * Z)

    # Do random walk
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = data.iloc[-1]
    for t in range(1, days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    #Load results into DataFrame
    montecarlo_df = pd.DataFrame(price_paths)

    #Average simulation results and make new dataframe for average AKA FORECAST
    montecarlo_avg = pd.DataFrame()
    montecarlo_avg[ticker] = montecarlo_df.mean(axis=1)


    #compute bollingerbands

    study = pd.DataFrame()
    study[ticker] = data[ticker]

    study = pd.concat([study, montecarlo_avg], axis=0, ignore_index=True)
    study_rows, study_cols = study.shape

    period = 25
    study['MA' + str(period)], study['upper'], study['lower'] = computeBollinger(study[ticker], period, std=2)

    # line 1 points
    x = study.index
    price = study[ticker]
    upperB = study['upper']
    lowerB = study['lower']
    MA = study['MA' + str(period)]
    # plotting the line 1 points
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    xmin, xmax, ymin, ymax = ax.axis()
    ax.plot(x, price, label=ticker)
    ax.plot(x, MA, label='price')
    ax.plot(x, upperB, label='upper')
    ax.plot(x, lowerB, label='lower')
    # ax.axvline(data_row, color="red", linestyle="--")

    data_row = len(data.index)

    buy_price = round(price.iloc[data_row], 2)

    take_win = round(upperB.iloc[-1] - price.iloc[data_row], 2)

    loss_price = round(((price.iloc[data_row] + MA.iloc[data_row]) / 2), 2)

    take_loss = round(loss_price - price.iloc[data_row], 2)

    win_price = round(upperB.iloc[-1], 2)
    print('===================================')
    print('Buy '+ ticker + ' @ $' + str(buy_price))
    print('Take Win @ $' + str(win_price))
    print('Stop Loss @ $' + str(loss_price))
    print('Possible Profit: $' + str(take_win))
    print('Possible Loss: $' + str(abs(take_loss)))
    print('REMINDER: take into account the leverage for ' + ticker)
    print('===================================')
    if abs(take_loss) > take_win:
      print('WARNING!!! Trade has greater potential loss than profit')
    else: pass

    r1 = mpatches.Rectangle((data_row, price.iloc[data_row]), days, take_win, color="green", alpha=0.20)
    r2 = mpatches.Rectangle((data_row, price.iloc[data_row]), days, take_loss, color="red", alpha=0.20)
    ax.add_patch(r1)
    ax.add_patch(r2)

    ax.axhline(price.iloc[data_row], color="grey", linestyle="--")
    ax.axhline(win_price, color="green", linestyle="--")
    ax.axhline(loss_price, color="red", linestyle="--")



    ax.axvline(data_row, color="grey", linestyle="--")

    #ax.autoscale_view('tight')
    
    ax.annotate('STOP LOSS: ' + str(round(loss_price, 2)), xy=(x[-1], loss_price), xycoords='data',
                xytext=(data_row, loss_price), textcoords='data',
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="rarrow,pad=0.3", fc="red", ec="b", lw=2)
                )
    ax.annotate('TAKE PROFIT: ' + str(round(win_price, 2)), xy=(x[-1], win_price), xycoords='data',
                xytext=(data_row, win_price), textcoords='data',
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="rarrow,pad=0.3", fc="green", ec="b", lw=2)
                )
    ax.annotate('BUY PRICE: ' + str(round(price.iloc[data_row], 2)), xy=(data_row, price.iloc[data_row]),
                xycoords='data',
                xytext=(data_row, price.iloc[data_row]), textcoords='data',
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
                )
    ax.grid(color='grey', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel( frequency + ' intervals between ' + str(start_date) + ' and current time on ' + str(today) + ' + trade horizon')
    # Set the y axis label of the current axis.
    ax.set_ylabel('price $')
    # Set a title of the current axes.
    ax.set_title(
        ticker + ' contract price from ' + str(start_date) + ' to ' + str(data.index[-1]) + '. Includes trade limit calculation')
    # show a legend on the plot
    ax.legend()
    # Display a figure.
    # ax.show()

  
calculate_limits('SOYB',10,'60m')
