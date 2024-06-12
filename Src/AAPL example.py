from articles_related import get_data, extract_relevant_articles
import yfinance as yf
import pandas as pd

#AAPL Example:

ticker = 'AAPL'
end = '2023-12-29'
start = '2020-03-28'

tickerData = yf.Ticker(ticker)

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start=start, end=end)

data_dict = tickerDf.to_dict(orient='index')

OCV_dict = {key.strftime('%Y-%m-%d'): {'Open': value['Open'], 'Close': value['Close'], 'Volume': value['Volume']}
                      for key, value in data_dict.items()} #example for a record in data_dict: '2019-12-30': {'Open': 70.41046286283884, 'Close': 70.91155242919922, 'Volume': 144114400}

data = get_data(ticker, start, end)

date_url_dict = extract_relevant_articles(ticker, data)

print(f" The date_url_dict: {date_url_dict} ")

# DataSets for evaluation:
short_interest_AAPL = pd.read_csv('Data/equityshortinterest_AAPL_NNM.csv')
oddlot_vol_AAPL = pd.read_csv('Data/oddlotvol_AAPL.csv')
change = short_interest_AAPL['% Change from Previous']
print(f"the change: {change.head}")
print(f"the oddlot_vol: {oddlot_vol_AAPL.head}")