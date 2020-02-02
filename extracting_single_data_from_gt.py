from pytrends.request import TrendReq

pytrend = TrendReq()


kw_list = ['enter singer']
pytrends = pytrend.build_payload(kw_list, cat=0, timeframe='2004-12-14 2010-12-25', geo='', gprop='')

pytrends = TrendReq(hl='en-US', tz=360)

interest_over_time_df = pytrend.interest_over_time()

interest_over_time_df.to_csv('enter path')
