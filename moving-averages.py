from pandas_datareader import data
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math

#Tickers we will be analyzing
tickers = ['AAPL','MSFT','SPY']
column_headers = ['date','open','high','low','close','volume']
datasource = 'iex'

start_date = '2013-01-01'
end_date = datetime.now()

#going to use IEX's api that gets pulled through pandas' datareader
panel_data = data.DataReader(tickers,datasource,start_date,end_date)
#returns a dict
#ix is depreceated in 0.22.0
#iex data gives you adjusted data

date_column = {}
for ticker in tickers:
    #get all data for ticker in tickers
    index = pd.DataFrame.from_dict(panel_data[ticker],orient = 'columns')

    date_column[ticker] = pd.DataFrame(index,columns = ['close']).rename(index = str, columns ={'close':ticker})
    if ticker == tickers[0]:
        ticker_table = date_column[ticker]
    else:
        #join faster than concat
        ticker_table = ticker_table.join(date_column[ticker])

#ticker_table.to_csv('tickers.csv',sep='\t', encoding='utf-8')

# Calculate the 20 and 100 days moving averages of the closing prices
short_rolling_ticker_of_interest = ticker_table.rolling(window=20).mean()
long_rolling_ticker_of_interest = ticker_table.rolling(window=100).mean()


#relative returns
returns = ticker_table.pct_change(1)
#log returns are calculated by taking the log of the interest and then diffing
log_returns = np.log(ticker_table).diff()
#cumulative log returns:
# c(t) = sum(r(t)) from 1 to t
# the relative return would be calculated as e^c(t) - 1

################################################################################################
#Simple Moving Average Calculation
################################################################################################
my_year_month_fmt = mdates.DateFormatter('%m/%y')

# Calculate the 20 and 100 days moving averages of the closing prices
stock_of_interest = 'MSFT'
SMA1 = 50
SMA2 = 200
_SMA1_day_simple_moving_average = ticker_table.rolling(window=SMA1).mean()
_SMA2_day_simple_moving_average = ticker_table.rolling(window=SMA2).mean()


################################################################################################
#Exponential Moving Average Calculation
################################################################################################

# Calculate the 20 and 100 days exponential moving averages of the closing prices
# Lag is calculated as M/2
N1 = 20
N2 = 26
_N1_day_exponential_moving_average = ticker_table.ewm(ignore_na = False,span = N1,min_periods = N1,adjust = True).mean()
_N2_day_exponential_moving_average = ticker_table.ewm(ignore_na = False,span = N2,min_periods = N2,adjust = True).mean()


################################################################################################
#Buy and Hold Strategy
################################################################################################
#Let's say we're going to hold these stocks with equal exposure
#N is our total portfolio value
#Exposure is the amount per stock (let's assume they are the same)
N = 100
Exposure = np.divide(1,len(tickers),dtype = float)
weights_matrix = pd.DataFrame(Exposure,index = ticker_table.index,columns = ticker_table.columns)

#approximation
#dot product of your weights and your stocks
log_returns_weighted_dot_product = weights_matrix.dot(log_returns.transpose())
portfolio_log_returns = pd.Series(np.diag(log_returns_weighted_dot_product), index=log_returns.index)
 #this is a first order taylor expansion of x= log(p(t))/log(p(t-1)) when x ~= 1
total_relative_returns = (np.exp(portfolio_log_returns.cumsum()) - 1)


#exact return
exact_relative_return = np.exp(log_returns)-1
exact_log_returns_weighted = weights_matrix * exact_relative_return
exact_total_relative_returns = exact_log_returns_weighted.cumsum().sum(axis=1)
print(exact_total_relative_returns[-1])
print(ticker_table)

# The last data point will give us the total portfolio return
# This could give us NaN if we don't do it correctly
total_portfolio_return = total_relative_returns[-1]
if math.isnan(total_portfolio_return):
    print("Warning: The most current day does not have data.")

# Average portfolio return assuming compunding of returns
number_of_years = float(end_date.year) - float(start_date.split('-',1)[0])
average_yearly_return = (1 + total_portfolio_return)**(1 / number_of_years) - 1
#Print total portfolio return
print('Total portfolio return is: ' +
      '{:5.2f}'.format(100 * total_portfolio_return) + '%')
print('Average yearly return is: ' +
      '{:5.2f}'.format(100 * average_yearly_return) + '%')
################################################################################################
#Strategy of EMAs
#difference of timeseries
timeseries_difference_ema = ticker_table - _N1_day_exponential_moving_average

#if EMA crosses price at time t, we will invert our strategy
EMA_weights_product = pd.DataFrame(weights_matrix.values*timeseries_difference_ema.apply(np.sign).values,columns=weights_matrix.columns,index=weights_matrix.index)

#since we don't know the timeseries difference until the end of day t, we have to right shift to correct for this lag
# Lagging our trading signals by one day.
EMA_trading_positions_final = EMA_weights_product.shift(1)

#approximate relative performance
EMA_dot_product= EMA_trading_positions_final.dot(log_returns.transpose())
portfolio_log_returns = pd.Series(np.diag(EMA_dot_product), index=log_returns.index)
approximate_total_relative_returns = (np.exp(portfolio_log_returns.cumsum()) - 1)


#real relative performance
EMA_relative_returns = np.exp(log_returns) - 1
EMA_trading_impact = EMA_relative_returns * EMA_trading_positions_final
EMA_relative_returns_cumsum = EMA_trading_impact.cumsum().sum(axis = 1)


# The last data point will give us the total portfolio return
# This could give us NaN if we don't do it correctly
total_portfolio_return = EMA_relative_returns_cumsum[-1]
if math.isnan(total_portfolio_return):
    print("Warning: The most current day does not have data.")

# Average portfolio return assuming compunding of returns
number_of_years = float(end_date.year) - float(start_date.split('-',1)[0])
average_yearly_return = (1 + total_portfolio_return)**(1 / number_of_years) - 1
#Print total portfolio return
print('Total portfolio return is: ' +
      '{:5.2f}'.format(100 * total_portfolio_return) + '%')
print('Average yearly return is: ' +
      '{:5.2f}'.format(100 * average_yearly_return) + '%')


################################################################################################
#Plots

#EMA
fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(1,1,1)
ax.plot(pd.to_datetime(_N1_day_exponential_moving_average[stock_of_interest].index),ticker_table[stock_of_interest],label=stock_of_interest)
ax.plot(pd.to_datetime(_N1_day_exponential_moving_average[stock_of_interest].index), _N1_day_exponential_moving_average[stock_of_interest], label= 'EMA ' + str(N1) +' days rolling')
ax.plot(pd.to_datetime(_N2_day_exponential_moving_average[stock_of_interest].index), _N2_day_exponential_moving_average[stock_of_interest], label='EMA '+ str(N2) +' days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()
#plt.show()


#SMA
fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(1,1,1)
ax.plot(pd.to_datetime(_SMA1_day_simple_moving_average[stock_of_interest].index),ticker_table[stock_of_interest],label=stock_of_interest)
ax.plot(pd.to_datetime(_SMA1_day_simple_moving_average[stock_of_interest].index), _SMA1_day_simple_moving_average[stock_of_interest], label= 'SMA '+ str(SMA1)+' days rolling')
ax.plot(pd.to_datetime(_SMA2_day_simple_moving_average[stock_of_interest].index), _SMA2_day_simple_moving_average[stock_of_interest], label='SMA '+str(SMA2)+' days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()
#plt.show()


#portfolio log returns given an exposure
fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(2, 1, 1)
ax.plot(pd.to_datetime(portfolio_log_returns.index), portfolio_log_returns.cumsum())
ax.set_ylabel('Portfolio cumulative log returns')
ax.grid()
ax = fig.add_subplot(2, 1, 2)
ax.plot(pd.to_datetime(total_relative_returns.index), 100 * total_relative_returns)
ax.set_ylabel('Portfolio total relative returns (%)')
ax.grid()
#plt.show()


# Plot everything by leveraging the very powerful matplotlib package
#this is just a time series plot with moving averages
fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(1,1,1)
for plot_ticker in ticker_table:
    ax.plot(pd.to_datetime(ticker_table[plot_ticker].index),ticker_table[plot_ticker],label=plot_ticker)
    ax.plot(pd.to_datetime(short_rolling_ticker_of_interest[plot_ticker].index), short_rolling_ticker_of_interest[plot_ticker], label= plot_ticker+ ' 20 days rolling')
    ax.plot(pd.to_datetime(long_rolling_ticker_of_interest[plot_ticker].index), long_rolling_ticker_of_interest[plot_ticker], label=plot_ticker + ' 100 days rolling')
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()
#plt.show()

fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(2,1,1)
for plot_ticker in ticker_table:
    ax.plot(pd.to_datetime(log_returns[plot_ticker].index),log_returns[plot_ticker].cumsum(0), label=plot_ticker)
ax.set_ylabel('Cumulative log returns')
ax.legend(loc='best')
ax.grid()
#plt.show()

fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(2,1,1)
for plot_ticker in ticker_table:
    ax.plot(pd.to_datetime(log_returns[plot_ticker].index),100*np.exp(log_returns[plot_ticker].cumsum(0))-1, label=plot_ticker)
ax.set_ylabel('Relative returns')
ax.legend(loc='best')
ax.grid()
#plt.show()
