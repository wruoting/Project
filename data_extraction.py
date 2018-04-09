from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


#Tickers we will be analyzing
tickers = ['AAPL','MSFT','SPY']
column_headers = ['date','open','high','low','close','volume']
datasource = 'iex'

start_date = '2013-01-01'
end_date = datetime.now()

#going to use IEX's api that gets pulled through pandas' datareader
panel_data = data.DataReader(tickers,datasource,start_date,end_date)

#create files for each ticker
for ticker in tickers:
    #get all data for ticker in tickers
    index = pd.DataFrame.from_dict(panel_data[ticker],orient = 'columns')
    index.to_csv('./TickerData/'+ticker+'.csv',sep='\t', encoding='utf-8')
