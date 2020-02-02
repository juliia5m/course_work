from pytrends.request import TrendReq

pytrend = TrendReq()
import pandas as pd

data = pd.read_csv('enter path', delimiter=',')
data = data.drop(data[(data.year < 2004)].index)  #cleaning dataset from unnecessary data

col = data.columns.tolist()
# d_a_s = data[[col[6],col[1]]]
author = data[col[7]].to_list()
song = data[col[2]].to_list()

pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), proxies=['https://46.151.145.4', ], retries=2,
                    backoff_factor=0.1)
for name in author:
    l = []
    for j in name:
        if j == "\\" or j == "/":
            name = name.replace(j, ",")
    l.append(name)
    kw_list = l

    pytrend.build_payload(kw_list, cat=0, timeframe='2004-12-14 2010-12-25', geo='', gprop='')

    interest_over_time_df = pytrend.interest_over_time()

    interest_over_time_df.to_csv('enter path' % (name))   #extracting data of authors from needed period and saving it to csv file
    #then use the same loop just for songs