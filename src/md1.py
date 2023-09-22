from cr import get_connection, get_ticker_dict
from st import get_price_table, get_volume_table
from matplotlib import pyplot
import pandas as pd
import sys

if __name__ == "__main__":
    con = get_connection() 
    q = "select * from kpfh where cnt = 4 and profit > 20"
    events = pd.read_sql(q, con=con)
    pdf = get_volume_table(con) 
    c = 0
    for index, val in events.iterrows():
        c += 1
        #if c > 30: break
        c = val["cnt"]
        s = val["sdate"] - pd.Timedelta(days=10)
        e = val["edate"] + pd.Timedelta(days=10)
        ticker = val["ticker"]

        v = pdf.loc[s:e, ticker]
        v *= 1000 / v.values[0]

        if v.max() > 20000: continue
    
        v.index = range(v.index.size)
        pyplot.plot(v)

        print(index, val)
        print(v)
   

    pyplot.show()
    
