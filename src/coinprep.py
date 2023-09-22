from cr import get_connection
import pandas as pd
from sys import exit
import numpy as np

def get_data(con, sym):
    start = "2021-02-01"
    end = "2023-09-10"

    q = f'''
    select * from ts_{sym.lower()}_m1 
    where "candleDateTime" >= '{start}' and "candleDateTime" < '{end}'
    order by  "candleDateTime" 
    '''

    return pd.read_sql(q, con=con)

def get_atr(df):
    df["atr1"] = df["highPrice"] - df["lowPrice"]

    df["yTradingPrice"] = None
    df["yTradingPrice"].iloc[1:] = df["tradePrice"].values[:-1]

    df["atr2"] = df["highPrice"] - df["yTradingPrice"]
    df["atr3"] = df["lowPrice"] - df["yTradingPrice"]

    df["atr"] = df.apply(lambda x: max( abs(x["atr1"]), abs(x["atr2"]), abs(x["atr3"]) ), axis=1)
    df["atr"] /= df["tradePrice"] / 100

    df["v"] = df["candleAccTradePrice"] / df["tradePrice"]
    df["tradePrice"] /= 1000
    t = df["timestamp"].values[0]
    df["timestamp"] -= t

    return df.iloc[1:, :]

if __name__ == "__main__":
    con = get_connection()

    df = get_data(con, 'BTC')

    df['change'] = df["tradePrice"].pct_change()
    df["change"] += 1
    df = get_atr(df)

    df = df[["tradePrice", "change", "atr", "v", "timestamp"]]
    np.save('data/btc.npy', df.values)
    print(df)
    print(df.describe())
