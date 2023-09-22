from pykrx import stock
from pykrx import bond
from sqlalchemy import create_engine
import pandas as pd
import sys
from sys import exit
import sqlalchemy

def get_connection():
    DB_HOST = "wwhale-on-gpt.cg6x7yqwsa6m.ap-northeast-2.rds.amazonaws.com"
    DB_USER = "postgres"
    DB_PASS = "a!02040608"
    DB_NAME = "systr"

    q= f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(q)
    return engine

def get_ticker_dict(date):
    tickers = stock.get_market_ticker_list("20230914")
    coms = []
    con = get_connection()

    code_dict = {}

    for ticker in tickers: 
        com = stock.get_market_ticker_name(ticker)
        code_dict[ticker] = com
    return code_dict

if __name__ == "__main__":

    dates = pd.date_range(sys.argv[1], sys.argv[2])[:-1]
    c = ["price", "marketcap", "volume", "volamount","shares"]

    for d in dates:

        df = stock.get_market_cap(d)
        df.columns = c
        df.index.name = 'ticker'
        df["date"] = d
        df = df[df['price'] != 0]

        if df.index.size == 0: continue
        
        tickers = stock.get_market_ticker_list(d)
        coms = []
        con = get_connection()
        print(len(tickers))

        code_dict = {}

        for ticker in tickers: 
            com = stock.get_market_ticker_name(ticker)
            code_dict[ticker] = com
            coms.append( ( ticker, com ) )
        
        def conv(n):
            try: 
                return code_dict[n]
            except KeyError:
                return ''

        df['ticker'] = df.index
        df['name'] = df.apply(lambda x: conv(x.name), axis=1)

        df = df[['ticker', 'date', 'name', *c]]
        df["marketcap"] /= 100000000
        print(df)
        df.to_sql('mkcap', if_exists='append', con = con, index=False)

    from sys import exit
    exit(0)
    
    c = [ 'open', 'high', 'low', 'end', 'volume', 'tot', 'change']
    for d in dates:
        df = stock.get_market_ohlcv(d,market="KOSPI")
        df = df[ df['시가'] != 0 ]

        df.columns = c

        df.insert(0, 'date','')
        df.insert(1, 'ticker','')
        df.insert(2, 'name','')

        df['date'] = d
        df['ticker'] = df.index
        df['name'] = df.apply(lambda x: conv(x.name), axis=1)
        print(df)
        try :
            df.to_sql('stpr', if_exists='append', con = con, index=False)
        except sqlalchemy.exc.IntegrityError:
            continue
