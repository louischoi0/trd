import pyupbit
from time import sleep
from sys import exit
import pandas as pd
import requests
import json
import numpy as np
from sqlalchemy import create_engine

def get_connection():
    DB_HOST = "wwhale-on-gpt.cg6x7yqwsa6m.ap-northeast-2.rds.amazonaws.com"
    DB_USER = "postgres"
    DB_PASS = "a!02040608"
    DB_NAME = "systr"

    q= f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(q)
    return engine

def getMinTicker(dt, currency, unit='1',candleunit='minutes', count=400) :
    dt = dt.strftime("%Y-%m-%d@%H:%M:00").replace("@","%20")
    endpoint = f"https://crix-api-endpoint.upbit.com/v1/crix/candles/{candleunit}/{unit}?code=CRIX.UPBIT.KRW-{currency}&count={count}&to={dt}"
    return endpoint
    return self.request(endpoint)


access = "Bkmm9gUIfuji0TwgzAikRul0L1Bx3Lxu6rBf5f3P"          # 본인 값으로 변경
secret = "WbthIqtWtzxyXIalwxRIdo0SRj74PxqfAJacQF0K"          # 본인 값으로 변경
upbit = pyupbit.Upbit(access, secret)

def order_buy(symbol, price, money):
    money *= 0.9995
    amount = money / price
    amount = round(amount, 8)

    rspn = upbit.buy_limit_order(f"KRW-{symbol}", price, amount)

    uuid = rspn["uuid"]
    print(f'order buy - symbol:{symbol} price:{price} amount:{amount} money:{amount*price}')
    return ( pd.Timestamp.now(), uuid, price, amount, amount*price )

prices = []

def get_market_price():
    p = pyupbit.get_current_price("KRW-BTC")
    return p

def get_buy_signal(p, prices, ma, w, o, p1th, p2th):
    #if len(prices) < w + o: return None
    p1 = prices[-o:]

    p2 = prices[-w-o: ]

    print(f'get buy signal - cp:{p} ')
    if p > ma and p1 > p1th and p2 > p2th:
        return p
    else:
        return None

def get_sell_signal(currentprice, position, h, losscut=0.975):
    buyts, uuid, buyprice, buyamount, _ = position

    profit = (currentprice - buyprice) / buyprice 
    profit *= 100
    now = pd.Timestamp.now()
    hts = now - buyts

    print(f'get sell signal - bp:{buyprice} cp:{currentprice} profit:{profit} bts:{buyts} hts: {hts}')

    #if hts > pd.Timedelta(minutes=3): # pd.Timedelta(minutes=h) :
    if hts > pd.Timedelta(minutes=h) :
        return True

    elif profit < -0.25:
        return True


    #return ( pd.Timestamp.now(), price, amount, amount*price )

def get_ma(days):
    df = pyupbit.get_ohlcv("KRW-BTC")
    mdf = df["close"].iloc[-days:]
    return mdf.mean()

if __name__ == "__main__":
    dt = pd.Timestamp.now()
    data = None
    con = get_connection()
    positions = []

    for i in range(7):
        e = getMinTicker(dt, 'BTC')
        dt -= pd.Timedelta(minutes=400)
        rspn = requests.get(e)
        d = json.loads(rspn.text)
        df = pd.DataFrame(d)

        if data is None :
            data = df
        else :
            data = pd.concat([data, df])

    data.index = data["candleDateTime"] 
    data = data.drop("candleDateTime", axis=1)
    data = data.sort_index()
    data = data.dropna()
    df = data
    prices =  data["tradePrice"].values
    lastmin = pd.Timestamp.now().minute
    symbol = 'BTC'

    while True:

        if len(prices) > 10000:
            prices = prices[100: ]

        ma = get_ma(10)

        p = get_market_price() 
    
        prices = np.append( prices, (p, ) )
        w, o, h = 1620, 660, 1900

        buyprice = get_buy_signal(p, prices, ma, w, o, 1.006, 1.015)

        if buyprice and len(positions) < 4:
            position = order_buy(symbol, p, 10000)
            uuid = position[1]
            positions.append(position)
            
            s = pd.DataFrame([(pd.Timestamp.now(), uuid, p, buyprice, 1000)])
            s.columns = ['time', 'uuid', 'p', 'buyprice', 'amount' ]
            s.to_sql('mmv3_buy_signal', con=con, if_exists='append', index=False)

        npositions = []
        
        for ps in positions:

            amount = ps[3]
            sellsignal = get_sell_signal(p, ps, h, losscut=0.975)
            print(f'sell - symbol:{symbol} amount:{amount} price:{p}')
            rspn = upbit.sell_limit_order(f"KRW-{symbol}", p, amount)
            uuid = rspn["uuid"]

            s = pd.DataFrame([(pd.Timestamp.now(), *ps, p)])
            s.to_sql('mmv3_sell_signal', con=con, if_exists='append', index=False)

            if not sellsignal: 
                npositions.append(ps)

        positions = npositions

        print(f'now ts:{pd.Timestamp.now()} price: {p} price len: {prices.shape}')

        while True:
            _lastmin = pd.Timestamp.now().minute
            if _lastmin != lastmin:
                lastmin = _lastmin
                break
            sleep(0.01)


